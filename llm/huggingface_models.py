from email import message
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os
import json
from huggingface_hub import snapshot_download
from typing import List

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
        added_pad_token = False
        
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
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                    added_pad_token = True
            if hasattr(self.tokenizer, "padding_side"):
                self.tokenizer.padding_side = "left"
            # ✅ Khi dùng quantization, phải force lên GPU (không thể dùng "auto" vì sẽ dispatch lên CPU/disk)
            device_map_value = "cuda:0" if torch.cuda.is_available() and use_quantization else "auto"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map_value,
                low_cpu_mem_usage=True,
                quantization_config=quantization_config,
                torch_dtype=torch_dtype,
            )
            if added_pad_token:
                self.model.resize_token_embeddings(len(self.tokenizer))
            if getattr(self.model.config, "pad_token_id", None) is None and self.tokenizer.pad_token_id is not None:
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
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
            
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                    added_pad_token = True
            if hasattr(self.tokenizer, "padding_side"):
                self.tokenizer.padding_side = "left"
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
            if added_pad_token:
                self.model.resize_token_embeddings(len(self.tokenizer))
            if getattr(self.model.config, "pad_token_id", None) is None and self.tokenizer.pad_token_id is not None:
                self.model.config.pad_token_id = self.tokenizer.pad_token_id

            
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

        # microsoft/phi-1_5 and similar leave eos/bos null in model config
        if getattr(self.model.config, "pad_token_id", None) is None and self.tokenizer.pad_token_id is not None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        if getattr(self.model.config, "eos_token_id", None) is None and self.tokenizer.eos_token_id is not None:
            self.model.config.eos_token_id = self.tokenizer.eos_token_id

        cfg_ctx = self.config.get("max_context_length")
        model_ctx = getattr(self.model.config, "max_position_embeddings", None)
        self._max_context_length = int(cfg_ctx or model_ctx or 8192)
        print(f"Max context length for generation: {self._max_context_length}")
        print("Model loaded with automatic device mapping across GPUs.")

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

    def _filter_messages(self, messages):
        """Drop empty system turns (needed for completion models like phi-1_5)."""
        filtered = []
        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            if role == "system" and (content is None or not str(content).strip()):
                continue
            filtered.append(message)
        return filtered

    def _build_chat_messages(self, system: str, user: str):
        messages = []
        if system and str(system).strip():
            messages.append({"role": "system", "content": str(system)})
        messages.append({"role": "user", "content": f"{user}"})
        return messages

    def _device(self):
        try:
            return self.model.device
        except Exception:
            return next(self.model.parameters()).device

    def _tokenize_for_generate(self, plain_text: str, max_new_tokens: int):
        inputs = self.tokenizer(plain_text, return_tensors="pt")
        inputs = {k: v.to(self._device()) for k, v in inputs.items()}
        max_total_length = self._max_context_length
        min_new_tokens = 16
        max_new_tokens = min(int(max_new_tokens), 4096)
        input_length = inputs["input_ids"].shape[-1]
        if input_length >= max_total_length:
            truncate_length = max(1, max_total_length - min_new_tokens)
            for k, v in inputs.items():
                if v.shape[-1] > truncate_length:
                    inputs[k] = v[:, -truncate_length:]
            max_new_tokens = min_new_tokens
        elif input_length + max_new_tokens > max_total_length:
            max_new_tokens = max(min_new_tokens, max_total_length - input_length)
        if max_new_tokens <= 0:
            max_new_tokens = min_new_tokens
        return inputs, max_new_tokens

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
        messages = self._build_chat_messages(system, user)
        try:
            plain_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            # Fallback: models that only accept user/model turns
            if messages and messages[0].get("role") == "system":
                system_text = messages[0]["content"]
                user_text = messages[1]["content"] if len(messages) > 1 else user
                messages = [{"role": "user", "content": f"{system_text}\n\n{user_text}"}]
            plain_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs, max_new_tokens = self._tokenize_for_generate(plain_text, max_length)
        gen_kwargs = {k: v for k, v in kwargs.items() if k not in ("max_new_tokens", "max_length")}

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **gen_kwargs,
        )
        response_start = inputs["input_ids"].shape[-1]
        response_ids = outputs[0][response_start:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return self._trim_stop_sequences(response)

    def generate_from_messages(self, messages, max_new_tokens, **kwargs):
        """
        Generate a response from a list of messages.
        """
        messages = self._filter_messages(messages)
        plain_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs, max_new_tokens = self._tokenize_for_generate(plain_text, max_new_tokens)
        gen_kwargs = {k: v for k, v in kwargs.items() if k not in ("max_new_tokens", "max_length")}

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **gen_kwargs,
        )

        response_start = inputs["input_ids"].shape[-1]
        response_ids = outputs[0][response_start:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return self._trim_stop_sequences(response)

    def generate_from_messages_batch(self, batch_messages, max_new_tokens, batch_size=2, **kwargs):
        """
        Generate a batch of responses from a list of messages.
        """
        plain_texts = []
        for messages in batch_messages:
            filtered = self._filter_messages(messages)
            plain_texts.append(self.tokenizer.apply_chat_template(filtered, tokenize=False, add_generation_prompt=True))
        inputs = self.tokenizer(plain_texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self._device()) for k, v in inputs.items()}
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        batch_size = input_ids.shape[0]

        max_total_length = self._max_context_length
        min_new_tokens = 16
        truncate_length = max_total_length - min_new_tokens
        max_new_tokens = min(max_new_tokens, 4096)

        row_id_lists = []
        row_truncated = []
        for i in range(batch_size):
            input_len = int(attention_mask[i].sum().item())
            ids = input_ids[i, :input_len].detach().cpu()
            if input_len >= max_total_length:
                ids = ids[-truncate_length:]
                row_truncated.append(True)
            else:
                row_truncated.append(False)
            row_id_lists.append(ids.tolist())
        
        padded = self.tokenizer.pad(
            {"input_ids": row_id_lists},
            return_tensors="pt",
            padding=True,
        )
        padded = {k: v.to(self._device()) for k, v in padded.items()}

        min_new_tokens_list = []
        for ids_list, truncated in zip(row_id_lists, row_truncated):
            len_ids = len(ids_list)
            if truncated:
                m_i = min_new_tokens
            else:
                m_i = max_new_tokens
                if len_ids + m_i > max_total_length:
                    m_i = max_total_length - len_ids
            if m_i <= 0:
                m_i = min_new_tokens
            min_new_tokens_list.append(m_i)

        max_new_tokens = min(min_new_tokens_list)

        gen_kwargs = {k: v for k, v in kwargs.items() if k not in ("max_new_tokens", "max_length")}

        outputs = self.model.generate(
            **padded,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **gen_kwargs,
        )
        input_length = padded["input_ids"].shape[-1]

        responses = []
        for i in range(outputs.shape[0]):
            response_ids = outputs[i][input_length:]
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            responses.append(self._trim_stop_sequences(response))
        return responses
    
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
        messages = self._filter_messages([
            {'role': 'system', 'content': f'{system}'},
            {'role': 'user', 'content': f'{user1}'},
            {'role': 'assistant', 'content': f'{assistant1}'},
            {'role': 'user', 'content': f'{user2}'},
        ])
        plain_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs, max_new_tokens = self._tokenize_for_generate(plain_text, max_length)
        gen_kwargs = {k: v for k, v in kwargs.items() if k not in ("max_new_tokens", "max_length")}

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **gen_kwargs,
        )
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
        messages = self._build_chat_messages(system, user)
        plain_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        plain_text += condition

        inputs, max_new_tokens = self._tokenize_for_generate(plain_text, max_length)
        gen_kwargs = {k: v for k, v in kwargs.items() if k not in ("max_new_tokens", "max_length")}

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **gen_kwargs,
        )
        response_start = inputs["input_ids"].shape[-1]
        response_ids = outputs[0][response_start:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return self._trim_stop_sequences(response)

    def conditional_generate_batch(self, conditions: List[str], systems: List[str], users: List[str], max_length: int = 1000, **kwargs):
        """
        Generate a batch of responses with additional conditions appended to the input prompt.
        """
        if not (len(conditions) == len(systems) == len(users)):
            raise ValueError(f"conditions, systems, and users must have same length, got {len(conditions)}, {len(systems)}, and {len(users)}")
        if len(users) == 0:
            return []
        plain_texts = []
        for condition, system, user in zip(conditions, systems, users):
            messages = self._build_chat_messages(system, user)
            plain_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            plain_text += condition
            plain_texts.append(plain_text)
        
        inputs = self.tokenizer(plain_texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self._device()) for k, v in inputs.items()}

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        batch_size = input_ids.shape[0]

        max_total_length = self._max_context_length
        min_new_tokens = 16
        truncate_length = max_total_length - min_new_tokens
        max_new_tokens = min(max_length, 4096)

        row_id_lists = []
        row_truncated = []

        for i in range(batch_size):
            input_len = int(attention_mask[i].sum().item())
            ids = input_ids[i, :input_len].detach().cpu()
            if input_len >= max_total_length:
                ids = ids[-truncate_length:]
                row_truncated.append(True)
            else:
                row_truncated.append(False)
            row_id_lists.append(ids.tolist())
        
        padded = self.tokenizer.pad(
            {"input_ids": row_id_lists},
            return_tensors="pt",
            padding=True,
        )

        padded = {k: v.to(self._device()) for k, v in padded.items()}

        min_new_tokens_list = []
        for ids_list, truncated in zip(row_id_lists, row_truncated):
            len_ids = len(ids_list)
            if truncated:
                m_i = min_new_tokens
            else:
                m_i = max_new_tokens
                if len_ids + m_i > max_total_length:
                    m_i = max_total_length - len_ids
            if m_i <= 0:
                m_i = min_new_tokens
            min_new_tokens_list.append(m_i)

        max_new_tokens = min(min_new_tokens_list)

        gen_kwargs = {k: v for k, v in kwargs.items() if k not in ("max_new_tokens", "max_length")}

        outputs = self.model.generate(
            **padded,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **gen_kwargs,
        )
        input_length = padded["input_ids"].shape[-1]

        responses = []
        for i in range(outputs.shape[0]):
            response_ids = outputs[i][input_length:]
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            responses.append(self._trim_stop_sequences(response))
        
        return responses

    def generate_batch(self, systems, users, max_length: int = 1000, **kwargs) -> List[str]:

        if len(systems) != len(users):
            raise ValueError(f"systems and users must have same length, got {len(systems)} and {len(users)}")
        if len(users) == 0:
            return []

        plain_texts = []
        for system, user in zip(systems, users):
            messages = self._build_chat_messages(system, user)
            plain_texts.append(self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

        inputs = self.tokenizer(plain_texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self._device()) for k, v in inputs.items()}

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        batch_size = input_ids.shape[0]
        max_total_length = self._max_context_length
        min_new_tokens = 16
        truncate_length = max(1, max_total_length - min_new_tokens)
        max_new_tokens = min(max_length, 4096)

        row_id_lists = []
        row_truncated = []
        for i in range(batch_size):
            input_len = int(attention_mask[i].sum().item())
            ids = input_ids[i, :input_len].detach().cpu()
            if input_len >= max_total_length:
                ids = ids[-truncate_length:]
                row_truncated.append(True)
            else:
                row_truncated.append(False)
            row_id_lists.append(ids.tolist())

        padded = self.tokenizer.pad(
            {"input_ids": row_id_lists},
            return_tensors="pt",
            padding=True,
        )
        padded = {k: v.to(self._device()) for k, v in padded.items()}

        min_new_tokens_list = []
        for ids_list, truncated in zip(row_id_lists, row_truncated):
            len_ids = len(ids_list)
            if truncated:
                m_i = min_new_tokens
            else:
                m_i = max_new_tokens
                if len_ids + m_i > max_total_length:
                    m_i = max_total_length - len_ids
            if m_i <= 0:
                m_i = min_new_tokens
            min_new_tokens_list.append(m_i)

        max_new_tokens = min(min_new_tokens_list)
        gen_kwargs = {k: v for k, v in kwargs.items() if k not in ("max_new_tokens", "max_length")}

        outputs = self.model.generate(
            **padded,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **gen_kwargs,
        )
        input_length = padded["input_ids"].shape[-1]

        responses = []
        for i in range(outputs.shape[0]):
            response_ids = outputs[i][input_length:]
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            responses.append(self._trim_stop_sequences(response))
        
        return responses

    def get_negative_log_likelihood(self, user_instruction: str, target_string: str):
        messages = self._build_chat_messages("You are a helpful assistant.", user_instruction)
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = self.tokenizer.encode(prompt_text, return_tensors="pt")
        target_ids = self.tokenizer.encode(target_string, add_special_tokens=False, return_tensors="pt")
        input_ids = torch.cat([prompt_ids, target_ids], dim=1).to(self._device())
        # Stay within model context (e.g. phi-1_5 = 2048)
        max_ctx = self._max_context_length
        if input_ids.shape[1] > max_ctx:
            overflow = input_ids.shape[1] - max_ctx
            # Keep as much of the target string as possible at the end
            prompt_keep = max(1, prompt_ids.shape[1] - overflow)
            prompt_ids = prompt_ids[:, -prompt_keep:]
            input_ids = torch.cat([prompt_ids, target_ids], dim=1)
            if input_ids.shape[1] > max_ctx:
                input_ids = input_ids[:, -max_ctx:]
                prompt_ids = input_ids[:, : max(1, input_ids.shape[1] - target_ids.shape[1])]
            input_ids = input_ids.to(self._device())
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

        labels = input_ids.clone()
        prompt_length = prompt_ids.shape[1]
        if prompt_length >= input_ids.shape[1]:
            prompt_length = max(1, input_ids.shape[1] - 1)
        labels[:, :prompt_length] = -100

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
        if outputs.loss is None:
            raise RuntimeError("Model returned no loss; check that labels has non-ignored positions.")
        
        return float(outputs.loss.item())