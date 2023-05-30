#!/usr/bin/env python
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
from model.unidm_base import UniDM


class UniDM_DataTransformation(UniDM):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        
        self.dataset_name = args.data_dir.split('/')[-1]

        self.dataset = args.dataset
        self.dataset_path = os.path.join(args.dataset_path, args.task, args.dataset)
        self.context_num = args.context_num
        self.Data_Parsing = args.data_parsing
        self.Prompt_Engineering = args.prompt_engineering
        self.dataset_name = args.data_dir.split('/')[-1]

    def get_prompt(self, table):
        """
        Generate the structured prompt.
        """
        columns = table.columns
        task_prompt = """Follow the example to transform the data:\n"""
        context=""
        cnt = 0
        series_major, series_c = table[columns[0]], table[columns[1]]

        for i,v in enumerate(series_major):
            if(cnt >= self.context_num):
                break
            context += """%s: %s\n%s: %s\n\n""" % (columns[0],v, columns[1],str(series_c[i]).strip('\t\t'))
            cnt += 1

        return task_prompt, context

    def data_parsing(self, task, context):
        prompt = "Summarize transformation pattern from text.\n\n"
        prompt += context+"\nTransformation pattern is:"
        gen_text = GPT_out(prompt=prompt,max_tokens=100)
        pattern_1 = gen_text.strip('\n')
        
        prompt = "Extract the specific transformation task from the text.\n\n"
        prompt += task+"\nTransformation task is:"
        gen_text = GPT_out(prompt=prompt,max_tokens=100)
        pattern_2 = gen_text.strip('\n')

        prompt = "Please summarize the final transformation pattern used for the given example based on the two patterns.\n"
        prompt += "Pattern 1: %s\nPattern 2: %s\nExample:\n%s"%(pattern_1,pattern_2,context)
        prompt += "The final correct transformation pattern is: "
        gen_text = GPT_out(prompt=prompt,max_tokens=100)
        output = gen_text.strip('\n')
        output = output.strip(' ')
        time.sleep(TIMESLEEP)
        return output

    def prompt_engineering(self, context, target):
        """
        Prompt engineering module.
        :param context: The context lines.
        :param target: The target line.
        """
        prompt = "Write the claim as the target text.\nClaim:\nThe context is\ndata before tansformation: 20000101\ndata after tansformation: 2000-01-01\ndata before tansformation: 20231220\ndata after tansformation: 2023-12-20\nThe target is\ndata before tansformation: 19990415\ndata after tansformation: \nTarget text:\n'20000101' to '2000-01-01'\n'20231220' to '2023-12-20'\n'19990415' to \n\nClaim:\n" 
        prompt += "The context is\n%s\n"%context.strip('\n')
        prompt += "The target is\n%s\n"%target
        prompt += "Target text:\n"
        gen_text = GPT_out(prompt=prompt,max_tokens=700)
        output = gen_text.strip('\n')
        time.sleep(TIMESLEEP)
        return output


    def UniDM(self, train, test, instruction):
        """
        Unified Framework for Data and Task with Large Language Models towards a Feature-rich Data Lake.
        :param tarin: The dataset to get the context.
        :param test: The dataset to test.
        :param instruction: The instruction for the data transformation dataset.
        """
        columns = train.columns
        head_c, head_major = columns[1], columns[0]
        
        task_prompt,context = self.get_prompt(train)
        # Parse data into a natural text representation
        if self.Data_Parsing:
            instruction = self.data_parsing(instruction,context)
                
        for i,line in test.iterrows():
            target_i = "%s: %s\n%s: " % (head_major,line[head_major],head_c)
            # Recursively uses the LLM to transform data tasks to the effective format
            if self.Prompt_Engineering:
                prompt_i = self.prompt_engineering(context,target_i)
            else:
                prompt_i = context + target_i

            prompt_i = instruction + "\n\n" + prompt_i
            gen_text = GPT_out(prompt=prompt_i,max_tokens=50)
            output = list(filter(None,gen_text.split('\n')))[0]
            print("Prediction: {}\nGround truth: {}\n".format(output,line[-1]))
            time.sleep(TIMESLEEP)


    def run(self):
        if self.dataset == "benchmark-stackoverflow":
            for x in os.listdir(self.dataset_path):
                file = open(os.path.join(self.dataset_path, x), 'r')
                lines = file.readlines()
                instruction = lines[0].split("//")[-1].strip("\n")
                if "txt" in x.split(".")[-1]: 
                    print(instruction)
                    file = pd.read_csv(os.path.join(self.dataset_path, x), sep="\t\t",  encoding='cp1252', 
                                    names=["data before tansformation", "data after tansformation"], index_col = False, skiprows=1, engine='python')   
                    self.UniDM(train=file[0:3], test=file[3:], instruction=instruction)
            
        elif self.dataset == "benchmark-bing-query-logs":
            for x in os.listdir(self.dataset_path):
                file = open(os.path.join(self.dataset_path, x), 'r')
                lines = file.readlines()
                instruction = lines[0].strip("\n")
                if "txt" in x.split(".")[-1] and "semantic" in x: 
                    print(instruction)
                    file = pd.read_csv(os.path.join(self.dataset_path, x), sep="\t\t",  encoding='cp1252', 
                                    names=["data before tansformation", "data after tansformation"], index_col = False, skiprows=1, engine='python')   
                    self.UniDM(train=file[0:3], test=file[3:], instruction=instruction)



if if __name__ == "__main__":
    model = UniDM_DataTransformation(args, logger)
    model.run()