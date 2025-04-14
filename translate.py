from transformers import MBart50Tokenizer, AutoModelForSeq2SeqLM, AutoTokenizer


class Translate:
    def __init__(self):
        self.tokenizer_vi2en = MBart50Tokenizer.from_pretrained("vinai/vinai-translate-vi2en-v2", src_lang="vi_VN")
        self.model_vi2en = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-vi2en-v2")
        self.tokenizer_en2vi = AutoTokenizer.from_pretrained("vinai/vinai-translate-en2vi-v2", src_lang="en_XX")
        self.model_en2vi = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-en2vi-v2")

    def vi2en(self,vi_text: str) -> str:
        input_ids = self.tokenizer_vi2en(vi_text, return_tensors="pt").input_ids
        output_ids = self.model_vi2en.generate(
            input_ids,
            decoder_start_token_id=self.tokenizer_vi2en.lang_code_to_id["en_XX"],
            num_return_sequences=1,
            num_beams=5,
            early_stopping=True
        )
        en_text = self.tokenizer_vi2en.batch_decode(output_ids, skip_special_tokens=True)
        en_text = " ".join(en_text)
        return en_text
    def en2vi(self,en_text: str) -> str:
        input_ids = self.tokenizer_en2vi(en_text, return_tensors="pt").input_ids
        output_ids = self.model_en2vi.generate(
            input_ids,
            decoder_start_token_id=self.tokenizer_en2vi.lang_code_to_id["vi_VN"],
            num_return_sequences=1,
            num_beams=5,
            early_stopping=True
        )
        vi_text = self.tokenizer_en2vi.batch_decode(output_ids, skip_special_tokens=True)
        vi_text = " ".join(vi_text)
        return vi_text