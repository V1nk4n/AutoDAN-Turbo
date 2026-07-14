from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os
import json
from huggingface_hub import snapshot_download

class HuggingFaceModel:
    def __init__(self, repo_name: str, config_dir: str, config_name: str, token=None, use_quantization=False, quantization_type="4bit"):
        """
        Initialize the Hugging Face model class in a distributed manner.

        Args:
            repo_name (str): Name of the Hugging Face model repository, e.g., "meta-llama/Meta-Llama-3-8B".
            config_dir (str): Directory where your config and template files are located.
            config_name (str): Name of the config file.
            token (str): Hugging Face API token for private models.
        """
        print(f"Checking for model in '{config_dir}/model_ckpt'...")
        model_dir = f"{config_dir}/model_ckpt"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, repo_name.replace("/", "_"))

        quantization_config = None
        torch_dtype = torch.float16
        
        if use_quantization:
            if quantization_type == "4bit":
                print("Using 4-bit quantization (INT4)...")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif quantization_type == "8bit":
                print("Using 8-bit quantization (INT8)...")
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                )
            torch_dtype = None  # Don't set torch_dtype when using quantization

        if not os.path.exists(model_path):
        #     print(f"Model not found in {model_path}. Downloading from Hugging Face...")
        #     AutoModelForCausalLM.from_pretrained(repo_name, token=token).save_pretrained(model_path)
        #     AutoTokenizer.from_pretrained(repo_name, token=token).save_pretrained(model_path)
        #     print(f"Model downloaded and saved to {model_path}.")
        # else:
        #     print(f"Model found in {model_path}. Using cached model.")
        # print(f"Loading model from {model_path}...")
            print(f"Model not found in {model_path}. Downloading model files directly (without loading into memory)...")
    
            # ✅ Dùng snapshot_download để download files trực tiếp
            print("Downloading model files from HuggingFace (this may take a while)...")
            snapshot_download(
                repo_id=repo_name,
                token=token,
                local_dir=model_path,
                local_dir_use_symlinks=False,
            )
            
            print(f"Model files downloaded to {model_path}.")
            print("Loading model with quantization...")
            
            # Load tokenizer và model từ local với quantization
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
            # ✅ Khi dùng quantization, phải force lên GPU (không thể dùng "auto" vì sẽ dispatch lên CPU/disk)
            device_map_value = "cuda:0" if torch.cuda.is_available() and use_quantization else "auto"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map_value,
                low_cpu_mem_usage=True,
                quantization_config=quantization_config,
                torch_dtype=torch_dtype,
            )
            print("Model loaded with quantization successfully!")
            
            # ✅ Clear CUDA cache sau khi load để giải phóng memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("CUDA cache cleared.")
        else:
            print(f"Model found in {model_path}. Loading from cache...")
    
            # ✅ Try load từ local, nếu fail thì load từ repo
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
                print("Tokenizer loaded from cache.")
            except Exception as e:
                print(f"Failed to load tokenizer from local: {e}")
                print("Re-downloading tokenizer from HuggingFace repo...")
                self.tokenizer = AutoTokenizer.from_pretrained(repo_name, token=token)
                self.tokenizer.save_pretrained(model_path)
                print("Tokenizer re-downloaded and saved.")
            
            # Load từ local với quantization
            # ✅ Khi dùng quantization, phải force lên GPU (không thể dùng "auto" vì sẽ dispatch lên CPU/disk)
            device_map_value = "cuda:0" if torch.cuda.is_available() and use_quantization else "auto"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map_value,
                low_cpu_mem_usage=True,
                quantization_config=quantization_config,
                torch_dtype=torch_dtype,
            )
            
            # ✅ Clear CUDA cache sau khi load để giải phóng memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("CUDA cache cleared.")


        self.config = json.load(open(f'{config_dir}/generation_configs/{config_name}.json'))

        # ✅ Ưu tiên custom chat_template từ file (relative to config_dir)
        chat_template_rel = (self.config.get("chat_template") or "").strip()
        chat_template_path = os.path.join(config_dir, chat_template_rel) if chat_template_rel else ""
        custom_template_loaded = False

        if chat_template_path and os.path.isfile(chat_template_path):
            try:
                # Load custom chat template từ file
                with open(chat_template_path, 'r', encoding='utf-8') as f:
                    chat_template = f.read()
                # Set chat template (không cần replace spaces vì jinja cần format)
                self.tokenizer.chat_template = chat_template
                custom_template_loaded = True
                print(f"✅ Loaded custom chat template from {chat_template_path}")
            except Exception as e:
                print(f"⚠️ Failed to load custom chat template from {chat_template_path}: {e}")


        # ✅ Fallback: dùng chat_template mặc định của tokenizer nếu custom không có
        if not custom_template_loaded:
            if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None:
                print("⚠️ Using tokenizer's built-in chat template (no custom template found)")
            else:
                print(f"⚠️ Warning: No chat template available (neither custom nor tokenizer default)")
        self._eos_token_ids = self._resolve_eos_token_ids()
        print(f"EOS token ids for generation: {self._eos_token_ids}")
        print("Model loaded with automatic device mapping across GPUs.")

    def _resolve_eos_token_ids(self):
        """Gemma needs both <eos> and <end_of_turn>; prefer model generation_config."""
        ids = []
        gen_cfg = getattr(self.model, "generation_config", None)
        if gen_cfg is not None and getattr(gen_cfg, "eos_token_id", None) is not None:
            eos = gen_cfg.eos_token_id
            ids = list(eos) if isinstance(eos, (list, tuple)) else [eos]
        if not ids and self.tokenizer.eos_token_id is not None:
            ids = [self.tokenizer.eos_token_id]
        for token_str in self.config.get("eos_extra_tokens", []) or []:
            try:
                tid = self.tokenizer.convert_tokens_to_ids(token_str)
            except Exception:
                tid = None
            if tid is None or tid == getattr(self.tokenizer, "unk_token_id", None):
                encoded = self.tokenizer.encode(token_str, add_special_tokens=False)
                tid = encoded[0] if len(encoded) == 1 else None
            if tid is not None and tid not in ids:
                ids.append(tid)
        # Gemma chat convention: stop at <end_of_turn> even if generation_config missing.
        for token_str in ("<end_of_turn>",):
            try:
                tid = self.tokenizer.convert_tokens_to_ids(token_str)
            except Exception:
                continue
            if tid is not None and tid != getattr(self.tokenizer, "unk_token_id", None) and tid not in ids:
                ids.append(tid)
        return ids if ids else self.tokenizer.eos_token_id

    def _trim_stop_sequences(self, text: str) -> str:
        stops = list(self.config.get("stop_sequences", []) or [])
        if not text or not stops:
            return text
        earliest = len(text)
        for stop in stops:
            if not stop:
                continue
            idx = text.find(stop)
            if idx != -1 and idx < earliest:
                earliest = idx
        return text[:earliest].rstrip() if earliest < len(text) else text

    def _merge_gen_kwargs(self, max_length: int, **kwargs):
        """Apply generation_config defaults, then caller overrides."""
        cfg = self.config or {}
        cfg_max = int(cfg.get("max_new_tokens", max_length))
        max_new_tokens = min(max_length, cfg_max, 4096)
        merged = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self._eos_token_ids,
        }
        temperature = float(cfg.get("temperature", 0.7))
        if "do_sample" not in kwargs:
            merged["do_sample"] = temperature > 0
        if merged.get("do_sample", kwargs.get("do_sample", False)):
            merged.setdefault("temperature", max(temperature, 1e-5))
            if "top_p" in cfg:
                merged.setdefault("top_p", float(cfg["top_p"]))
            if "top_k" in cfg:
                merged.setdefault("top_k", int(cfg["top_k"]))
        if "repetition_penalty" in cfg:
            merged.setdefault("repetition_penalty", float(cfg["repetition_penalty"]))
        if "no_repeat_ngram_size" in cfg:
            merged.setdefault("no_repeat_ngram_size", int(cfg["no_repeat_ngram_size"]))
        merged.update(kwargs)
        if not merged.get("do_sample", False):
            merged.pop("temperature", None)
            merged.pop("top_p", None)
            merged.pop("top_k", None)
        return merged

    def generate(self, system: str, user: str, max_length: int = 1000, **kwargs):
        """
        Generate a response based on the input text.

        Args:
            system (str): System message for the model.
            user (str): User message for the model.
            max_length (int): Maximum length of the generated response.
            **kwargs: Additional optional parameters such as temperature, top_k, top_p.

        Returns:
            str: The generated response from the model.
        """
        messages = []
        # Gemma and some chat templates reject empty/unsupported system turns.
        if system and str(system).strip():
            messages.append({'role': 'system', 'content': str(system)})
        messages.append({'role': 'user', 'content': f'{user}'})
        try:
            plain_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            # Fallback: models that only accept user/model turns (older Gemma).
            if messages and messages[0].get('role') == 'system':
                system_text = messages[0]['content']
                user_text = messages[1]['content'] if len(messages) > 1 else user
                messages = [{'role': 'user', 'content': f"{system_text}\n\n{user_text}"}]
            plain_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Model and tokenizer will handle device placement automatically
        inputs = self.tokenizer(plain_text, return_tensors="pt")
        # Move inputs to the correct device based on their device_map
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen_kwargs = self._merge_gen_kwargs(max_length, **kwargs)
        outputs = self.model.generate(**inputs, **gen_kwargs)
        response_start = inputs["input_ids"].shape[-1]
        response_ids = outputs[0][response_start:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return self._trim_stop_sequences(response)

    def continue_generate(self, system: str, user1: str, assistant1: str, user2: str, max_length: int = 1000, **kwargs):
        """
        Continue a conversation and generate a response.

        Args:
            system (str): System message for the model.
            user1 (str): User message for the model.
            assistant1 (str): Assistant message for the model.
            user2 (str): User message for the model.
            max_length (int): Maximum length of the generated response.
            **kwargs: Additional optional parameters such as temperature, top_k, top_p.

        Returns:
            str: The generated response from the model.
        """
        messages = [
            {'role': 'system', 'content': f'{system}'},
            {'role': 'user', 'content': f'{user1}'},
            {'role': 'assistant', 'content': f'{assistant1}'},
            {'role': 'user', 'content': f'{user2}'},
        ]
        plain_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer(plain_text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen_kwargs = self._merge_gen_kwargs(max_length, **kwargs)
        outputs = self.model.generate(**inputs, **gen_kwargs)
        response_start = inputs["input_ids"].shape[-1]
        response_ids = outputs[0][response_start:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return self._trim_stop_sequences(response)

    def conditional_generate(self, condition: str, system: str, user: str, max_length: int = 1000, **kwargs):
        """
        Generate a response with additional conditions appended to the input prompt.

        Args:
            condition (str): Condition for the generation (appended to the prompt).
            system (str): System message for the model.
            user (str): User message for the model.
            max_length (int): Maximum length of the generated response (treated as max_new_tokens).
            **kwargs: Additional optional parameters such as temperature, top_k, top_p.

        Returns:
            str: The generated response from the model.
        """
        messages = [
            {'role': 'system', 'content': f'{system}'},
            {'role': 'user', 'content': f'{user}'},
        ]
        plain_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        plain_text += condition

        inputs = self.tokenizer(plain_text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Calculate input length
        input_length = inputs["input_ids"].shape[-1]
        
        # Use max_new_tokens instead of max_length to avoid conflicts
        # max_length is treated as max_new_tokens (tokens to generate)
        # But we need to ensure it doesn't exceed model's context window
        max_new_tokens = min(max_length, 4096)  # Cap at 4096 for safety
        
        # If input is already too long, reduce max_new_tokens proportionally
        # Most models have context window of 8192 or more
        # We'll allow up to 8192 total tokens (input + output)
        max_total_length = 8192
        min_new_tokens = 16  # Minimum tokens to generate (fallback value)
        
        if input_length >= max_total_length:
            # Input is too long, truncate it to make room for output
            # Keep the most recent part of the input (tail) and reserve space for output
            truncate_length = max_total_length - min_new_tokens
            for k, v in inputs.items():
                if v.shape[-1] > truncate_length:
                    inputs[k] = v[:, -truncate_length:]
                    input_length = truncate_length
            max_new_tokens = min_new_tokens
        elif input_length + max_new_tokens > max_total_length:
            max_new_tokens = max_total_length - input_length
        
        # Final safety check: ensure max_new_tokens is always > 0
        if max_new_tokens <= 0:
            max_new_tokens = min_new_tokens

        gen_kwargs = self._merge_gen_kwargs(max_new_tokens, **kwargs)
        # Respect context-window clamp computed above.
        gen_kwargs["max_new_tokens"] = max_new_tokens
        outputs = self.model.generate(**inputs, **gen_kwargs)
        response_start = inputs["input_ids"].shape[-1]
        response_ids = outputs[0][response_start:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return self._trim_stop_sequences(response)
