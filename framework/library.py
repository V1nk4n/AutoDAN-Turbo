import os
import json

class Library():
    def __init__(self, library, logger):
        self.library = library
        self.logger = logger

    def merge(self, dict1, dict2):
        if not dict1:
            return dict2
        merged = {}
        all_keys = set(dict1.keys()) | set(dict2.keys())
        for key in all_keys:
            if key in dict1 and key in dict2:
                merged[key] = dict1[key]
                if "Example" in dict1[key] and "Example" in dict2[key] and dict2[key]["Example"]:
                    dict1[key]["Example"].append(dict2[key]["Example"][0])
                    dict1[key]["Score"].append(dict2[key]["Score"][0])
                    dict1[key]["Embeddings"].append(dict2[key]["Embeddings"][0])
            elif key in dict1:
                merged[key] = dict1[key]
            else:
                merged[key] = dict2[key]
        return merged

    def add(self, new_strategy, if_notify=False):
        '''
        :param new_strategy: a dictionary containing the new strategy to be added to the library
        '''
        # Validate required keys
        if "Strategy" not in new_strategy:
            raise KeyError(f"Missing required key 'Strategy' in new_strategy. Got keys: {list(new_strategy.keys())}")
        if "Definition" not in new_strategy:
            raise KeyError(f"Missing required key 'Definition' in new_strategy. Got keys: {list(new_strategy.keys())}")
        
        # Handle case where "Strategy" might be a list instead of string
        strategy_value = new_strategy["Strategy"]
        if isinstance(strategy_value, list):
            # If Strategy is a list, take the first element or join them
            if len(strategy_value) > 0:
                strategy_value = strategy_value[0] if isinstance(strategy_value[0], str) else str(strategy_value[0])
            else:
                strategy_value = "Unknown Strategy"
            # Update the strategy value in the dictionary
            new_strategy["Strategy"] = strategy_value
            if self.logger:
                self.logger.warning(f"Strategy was a list, converted to string: {strategy_value}")
        elif not isinstance(strategy_value, str):
            # Convert to string if it's not a string
            strategy_value = str(strategy_value)
            new_strategy["Strategy"] = strategy_value
            if self.logger:
                self.logger.warning(f"Strategy was not a string, converted: {strategy_value}")
        
        new_strategy = {strategy_value: new_strategy}
        self.library = self.merge(self.library, new_strategy)
        if if_notify:
            for key, value_dict in new_strategy.items():
                new_dict = {"Strategy": value_dict["Strategy"], "Definition": value_dict["Definition"]}
                self.logger.info(f'New strategy added: {json.dumps(new_dict, indent=4, ensure_ascii=False)}')

    def all(self):
        return self.library
