#!/usr/bin/env python
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
from model.unidm_base import UniDM


class UniDM_DataTransformation(UniDM):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        self.manifest.stop_token = '\n\n'
        self.dataset_name = args.data_dir.split('/')[-1]
        self.pe_suffix = "Follow the example to transform the data:\n"

    def data_parsing(self, task, context):
        prompt = "Summarize transformation pattern from text.\n\n"
        prompt += context + "\nTransformation pattern is:"
        gen_text = self.apply_prompt(prompt=prompt)
        pattern_1 = gen_text.strip('\n')
        
        prompt = "Extract the specific transformation task from the text.\n\n"
        prompt += task + "\nTransformation task is:"
        gen_text = self.apply_prompt(prompt=prompt)
        pattern_2 = gen_text.strip('\n')

        prompt = "Please summarize the final transformation pattern used for the given example based on the two patterns.\n"
        prompt += "Pattern 1: %s\nPattern 2: %s\nExample:\n%s"%(pattern_1,pattern_2,context)
        prompt += "The final correct transformation pattern is: "
        gen_text = self.apply_prompt(prompt=prompt)
        output = gen_text.strip('\n')
        output = output.strip(' ')
        return output

    def prompt_engineering(self, context, target):
        """
        Prompt engineering module.
        :param context: The context lines.
        :param target: The target line.
        """
        prompt = "Write the claim as target text.\nClaim:\nThe context is\ndata before tansformation: 20000101\ndata after tansformation: 2000-01-01\ndata before tansformation: 20231220\ndata after tansformation: 2023-12-20\nThe target query is\ndata before tansformation: 19990415\ndata after tansformation: \nTarget text:\n20000101 to 2000-01-01\n20231220 to 2023-12-20\n19990415 to \n\n"
        prompt += "Claim:\n" 
        prompt += "The context is\n%s\n"%context.strip('\n')
        prompt += "The target query is\n%s\n"%target
        prompt += "Target text:\n"
        gen_text = self.apply_prompt(prompt=prompt)
        output = gen_text.strip('\n')
        return output

    def run(self, train_data, test_data):
        preds = []

        for i,row in test_data.iterrows():
            target_Q =  f"data before tansformation: {row['input']}\ndata after tansformation: "

            # Parse data into a natural text representation
            if self.Data_Parsing:
                instruction = self.data_parsing(row['instruction'], row['context'])
            else:
                instruction = row['instruction']

            # Recursively uses the LLM to transform data tasks to the effective format
            if self.Prompt_Engineering:
                prompt_as = self.prompt_engineering(row['context'], target_Q)
            else:
                prompt_as = row['context'] + target_Q

            prompt_as = instruction + "\n\n" + prompt_as
        
            gen_text = self.apply_prompt(prompt=prompt_as)
            pred = list(filter(None,gen_text.split('\n')))[0]

            self.logger.info("ID: {} => Prediction: {}. Ground truth: {}. \n".format(i, pred, row['label_str'].strip()))
            
            preds.append(pred)
            self.p_as.append(prompt_as)

        return preds
