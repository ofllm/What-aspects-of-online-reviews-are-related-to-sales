import json
import torch
import transformers


class Sentiment:
    @staticmethod
    def encoder():

        MODEL_PATH = r"../model/chinese_roberta_wwm_large_ext_pytorch"
        # 导入模型
        tokenizer = transformers.BertTokenizer.from_pretrained(
            r"../model/chinese_roberta_wwm_large_ext_pytorch/vocab.txt")
        # 导入配置文件
        model_config = transformers.BertConfig.from_pretrained(MODEL_PATH)
        # 修改配置
        model_config.output_hidden_states = True
        model_config.output_attentions = True

        # 通过配置和路径导入模型
        model = transformers.BertModel.from_pretrained(MODEL_PATH, config=model_config)

        sentences = ["品质", "口味", "新鲜", "份量", "大小", "包装", "价格", "客服", "快递", "回购", "推荐"]
        sentences_token = tokenizer(sentences, padding=True, return_tensors='pt')

        output = model(input_ids=sentences_token['input_ids'], attention_mask=sentences_token['attention_mask'])

        return output




if __name__ == '__main__':
    output = Sentiment.encoder()[1]
    print()
