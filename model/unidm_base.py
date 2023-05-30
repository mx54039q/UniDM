#!/usr/bin/env python
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
import os, time
from manifest import Manifest


TIMESLEEP = 1


class UniDM():
    def __init__(self, args, logger):
        self.context_num = args.context_num
        self.instance_num = args.instance_num
        self.instance_wise = args.instance_wise
        self.metadata_wise = args.metadata_wise
        self.Data_Parsing = args.data_parsing
        self.Prompt_Engineering = args.prompt_engineering
        self.logger = logger
        self.manifest = Manifest(
            client_name='openai',
            cache_name='sqlite',
            cache_connection='unifdt.sqlite',
            stop_token='\n',
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=1.0,
            n=1,
        )

        self.p_as = []
        self.score_table = []
        self.total_num_toks = 0

    def apply_prompt(self, prompt):
        res = self.manifest.run(prompt=prompt)
        self.total_num_toks += len(prompt) // 4
        time.sleep(TIMESLEEP)
        return res

    def get_fee(self):
        return self.total_num_toks*0.02/1000

    def get_prompt(self, table):
        """
        Generate the prompt.
        """
        raise NotImplementedError("")

    def metadata_retrieval(self, table):
        """
        The metadata-wise component of the auto-retrieve module.
        """
        raise NotImplementedError("")

    def instance_retrieval(self, train, test, i):
        """
        The Instance-wise component of the auto-retrieve module.
        """
        raise NotImplementedError("")

    def data_parsing(self, context):
        """
        Adaptive data parsing module.
        """
        raise NotImplementedError("")

    def prompt_engineering(self, context, target):
        """
        Prompt engineering module.
        :param context: The context lines.
        :param target: The target line.
        """
        raise NotImplementedError("")

    def run(self, train, test):
        """
        Unified Framework for Data and Task with Large Language Models towards a Feature-rich Data Lake.
        :param tarin: The dataset to get the context.
        :param test: The dataset to test.
        """
        raise NotImplementedError("")



