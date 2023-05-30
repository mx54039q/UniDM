#!/usr/bin/env python
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
import pandas as pd

from model.unidm_base import UniDM
from utils.constants import IMPUTE_COLS
from utils.data_utils import serialize_row


class UniDM_DataImputation(UniDM):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        self.prompt_dp = "Given the items and convert the them into a textual format in a logical order.\n The items are %s\n"

        self.dataset_name = args.data_dir.split('/')[-1]
        self.impute_col = constants.IMPUTE_COLS[dataset_name]
        self.score_table = []

    def metadata_retrieval(self, table):
        """
        The metadata-wise component of the auto-retrieve module.
        """
        columns = table.columns
        attributes = ["%s(id:%s)"%(c,i) for i,c in enumerate(columns) if c != self.major_c and c != self.impute_col and c != 'label_str']

        prompt_rm = "The task is data imputation. The target query is '%s'.\n" % self.impute_col
        prompt_rm += "The attributes about '%s' are %s.\n"%(self.dataset_name, attributes)
        prompt_rm += "Which attribute is the most helpful for task and query?\nGive me ID only: "
        gen_text = self.apply_prompt(prompt=prompt_rm)

        output = list(filter(None, gen_text.split(' ')))[0]
        # output = list(filter(None, gen_text.split('\n')))[0]
        return int(output)

    def instance_retrieval(self, candidates, target_Q, i, column_map):
        """
        The Instance-wise component of the auto-retrieve module.
        """
        if len(self.score_table) != 0:
            table = candidates.copy()
            table["score"] = score_table[i]
            table.sort_values(by=["score"],axis=0,ascending=False,inplace=True)
            table.reset_index(drop=True,inplace=True)
            table = table.drop('score',axis=1)
            retrieved_table = table.iloc[:self.instance_num]
            return retrieved_table

        score = []
        prompt_ri_prefix = """The task is data imputation.\nThe target query is '%s'.\n"""%(target_Q)
        for _, instance in candidates.iterrows():

            ins_serialized = serialize_row(instance, column_map) + ". %s:%s" % (self.impute_col, instance[self.impute_col])

            prompt_ri = prompt_ri_prefix + """The give instance is '%s'\n"""%(ins_serialized)
            prompt_ri += """Score the relevance of give instance to target query.\n"""
            prompt_ri += """Use the following scoring system:\n0 - Not relevant at all\n1 - Slightly relevant \n2 - Moderately relevant\n3 - Highly relevant\n\n"""
            prompt_ri += """(0/1/2/3):"""

            gen_text = self.apply_prompt(prompt=prompt_ri)
            output = list(filter(None,gen_text.split('\n')))[0]
            score.append(int(output))

        self.score_table.append(score)

        table = candidates.copy()
        table["score"] = score 
        table.sort_values(by=["score"],axis=0,ascending=False,inplace=True)
        table.reset_index(drop=True,inplace=True)
        table = table.drop('score',axis=1)
        retrieved_table = table.iloc[:self.instance_num]
        return retrieved_table

    def data_parsing(self, context):
        """
        Adaptive data parsing module.
        """
        prompt = self.prompt_dp % context
        gen_text = self.apply_prompt(prompt=prompt)
        # output = gen_text.strip('\n')
        return gen_text

    def prompt_engineering(self, context, target):
        """
        Prompt engineering module.
        :param context: The context lines.
        :param target: The target line.
        """
        if self.dataset_name == "Restaurant":
            prompt_cq = "Write the claim as a clozen question.\n\nClaim:\nThe context is Wenham, Marysville, and Westmont are cities in the United States, identified by the ISO3 code USA. The target is city: New Cassel iso3: USA country: __\nCloze question:\nWenham, Marysville, and Westmont are cities in the United States, identified by the ISO3 code USA.\nNew Cassel is the name of a city whose ISO3 country code is USA. New Kassel belongs to the country __.\n\n" 
        else:
            prompt_cq = "Write the claim as a clozen question.\n\nClaim:\nThe context is The Griffin Protective Wave Case for Smart Phone - 8227-IP2WVB is a black case designed for the iPhone 3G, manufactured by Griffin. The target is name: Panasonic KX-TCA86 Headset description: Over-the-head manufacturer: __\nClozen question:\nThe manufacturer is __. [Pure Digital Technol,LG Electronics,ELGATO SYSTEMS,Samsung,Monster]\nThe Griffin Protective Wave Case for Smart Phone - 8227-IP2WVB is a black case designed for the iPhone 3G. The manufacturer is __.[Griffin]\nThe Panasonic KX-TCA86 Headset is an over-the-head headset. The manufacturer is __.[]\n\n" 

        prompt_cq += "Claim:\n"
        prompt_cq += "The context is %s "%context
        prompt_cq += "The target is %s\n"%target
        prompt_cq += "Cloze question:\n"

        # print(prompt)
        gen_text = self.apply_prompt(prompt=prompt_cq)
        output = gen_text.strip('\n')
        return output

    def run(self, train_data, test_data):
        """
        :param train_data: The dataset to get the context.
        :param test_data: The dataset to test.
        """

        self.major_c = train_data.columns[0]

        # Metadata-wise retrieve
        if self.metadata_wise:
            column_map = {self.major_c:self.major_c}
            metadata_c_idx = self.metadata_retrieval(train_data)
            metadata_c = train_data.columns[metadata_c_idx]
            column_map[metadata_c] = metadata_c
        else:
            column_map = {c: c for c in train_data.columns if c != "id" and c != self.impute_col and c != 'label_str'}

        # Load instance-retrieval result if exists
        score_table_name = "dataset%s_candidate%d_ins%d.npy" % (self.dataset_name, self.context_num, self.instance_num)
        if os.path.exists(score_table_name):
            self.score_table = np.load(score_table_name)

        # 
        preds = []
        for i,row in test_data.iterrows():
            # Query 
            row_serialized = serialize_row(row, column_map)
            target_Q = row_serialized + ". " + "%s: __" % self.impute_col

            # Instance-wise retrieve
            if self.instance_wise:
                instance_candidate = train_data.sample(self.context_num)
                retrieved_table = self.instance_retrieval(instance_candidate, target_Q, i, column_map)
                context = retrieved_table.apply(
                    lambda row: serialize_row(row,column_map),
                    axis=1,
                )
            else:
                random_instances = train_data.sample(self.instance_num)
                context = random_instances.apply(
                    lambda row: serialize_row(row,column_map),
                    axis=1,
                )
            
            # Parse data into a natural text representation
            context = list(context)
            if self.Data_Parsing:
                context = [model.data_parsing(c) for c in context]
            context = ' '.join(context)

            # Recursively uses the LLM to transform data tasks to the effective format
            if self.Prompt_Engineering:
                prompt_as = self.prompt_engineering(context, target_Q)
                prompt_as += "\nAnswer:"
            else:
                if self.Data_Parsing:
                    prompt_as = context + "\n" + target_Q + "\nAnswer:"
                else:
                    prompt_as = "Follow the example to impute the missing value.\n" + context + target_Q +"\nAnswer:"

            gen_text = self.apply_prompt(prompt=prompt_as)
            pred = list(filter(None,gen_text.split('\n')))[0]
            pred = pred.strip('\n')
            print("ID: {} => Prediction: {}. Ground truth: {}. \n".format(i, pred, row['label_str'].strip()))
            
            preds.append(pred)

        # Save the score table
        if self.instance_wise:
            if not os.path.exists(score_table_name):
                np.save(score_table_name, np.array(self.score_table))

        return preds


if __name__ == "__main__":
    model = UniDM_DataImputation(args, logger)
