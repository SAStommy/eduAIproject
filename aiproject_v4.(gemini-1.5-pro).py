import gradio as gr
import google.generativeai as genai
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from huggingface_hub import login
import re
import csv
import chardet
from google.generativeai.types import HarmCategory, HarmBlockThreshold

#audio ai import
login(input("your hugging face key?"))
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3")
processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
model.to(device)

audioai = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    torch_dtype=torch_dtype,
    device=device,
)

#gemini import
genai.configure(api_key=input("your gemini key?"))
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel('gemini-1.5-pro',  generation_config=generation_config, safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    })

#本機端audio需要ffmpeg
#分割跑了2000s, 不分割1500s(5:30)
#400s(2:00)
#260s(1:20)

# learning_area part v3
def remove_mark(text):
    # 移除HTML標記
    text = re.sub(r'<[^>]+>', '', text)
    text = text.replace('#', '')
    text = text.replace('*', '')
    return text

def csv_to_list(csv_file):
    result_list = []
    with open(csv_file, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
    
    with open(csv_file, 'r', newline='', encoding=encoding) as file:
        reader = csv.DictReader(file)
        count = 0
        for row in reader:
            if count < 100:
                result_list.append(f"input: {row['input:']}")
                result_list.append(f"output: {row['output:']}")
                count += 1
            else:
                break
    #print(result_list)
    return result_list


traindata = csv_to_list("finaltrain.csv")


def main(audio, course, wordlimit, feeling, AImark, mark):
    
    if mark > 50:
        type = "正面"
    else:
        type = "負面"
        
    
    #audio part v2
    if audio and course and wordlimit > 0 and feeling and AImark:
        audiocontent = audioai(audio)
        response = model.generate_content(traindata + ["input: " + audiocontent["text"] + f",請根據上述影片內容生成{course}課程的{feeling}心得並限制在{wordlimit}字內", "output: "])
        askmark = model.generate_content(response.candidates[0].content.parts[0].text + "請根據上文課程描述為課程評分, 只需回應1-100的數字")
        mark = askmark.candidates[0].content.parts[0].text
        learning_area = model.generate_content(response.candidates[0].content.parts[0].text + "請把大學常用單字翻譯成英文, 若文本為英文則翻譯成中文, 並以第一行為*原文單字:翻譯單字*及第二行為*例句: 翻譯單字例句並換行2次格式") 
        return remove_mark(response.candidates[0].content.parts[0].text), mark, remove_mark(learning_area.candidates[0].content.parts[0].text)
    elif audio and course and wordlimit > 0 and feeling and mark != -1:
        audiocontent = audioai(audio)
        response = model.generate_content(traindata + ["input: " + audiocontent["text"] + f",請根據上述影片內容生成請生成{course}課程的{feeling}且符合該{mark}分(滿分為100分)的{type}心得並限制在{wordlimit}字內", "output: "])
        learning_area = model.generate_content(response.candidates[0].content.parts[0].text + "請把大學常用單字翻譯成英文, 若文本為英文則翻譯成中文, 並以第一行為*原文單字:翻譯單字*及第二行為*例句: 翻譯單字例句並換行2次格式") 
        return remove_mark(response.candidates[0].content.parts[0].text), mark, remove_mark(learning_area.candidates[0].content.parts[0].text)
    
    elif audio and wordlimit > 0 and feeling and mark != -1:
        audiocontent = audioai(audio)
        response = model.generate_content(traindata + ["input: " + audiocontent["text"] + f",請根據上述影片內容生成請生成課程的{feeling}且符合該{mark}分(滿分為100分)的{type}心得並限制在{wordlimit}字內", "output: "])
        learning_area = model.generate_content(response.candidates[0].content.parts[0].text + "請把大學常用單字翻譯成英文, 若文本為英文則翻譯成中文, 並以第一行為*原文單字:翻譯單字*及第二行為*例句: 翻譯單字例句並換行2次格式") 
        return remove_mark(response.candidates[0].content.parts[0].text), mark, remove_mark(learning_area.candidates[0].content.parts[0].text)
    elif audio and course and wordlimit > 0 and mark != -1:
        audiocontent = audioai(audio)
        response = model.generate_content(traindata + ["input: " + audiocontent["text"] + f",請根據上述影片內容生成{course}課程的且符合該{mark}分(滿分為100分)的{type}心得並限制在{wordlimit}字內", "output: "])
        learning_area = model.generate_content(response.candidates[0].content.parts[0].text + "請把大學常用單字翻譯成英文, 若文本為英文則翻譯成中文, 並以第一行為*原文單字:翻譯單字*及第二行為*例句: 翻譯單字例句並換行2次格式") 
        return remove_mark(response.candidates[0].content.parts[0].text), mark, remove_mark(learning_area.candidates[0].content.parts[0].text)
    elif audio and course and feeling and mark != -1: 
        audiocontent = audioai(audio)
        response = model.generate_content(traindata + ["input: " + audiocontent["text"] + f",請根據上述影片內容生成{course}課程的且符合該{mark}分(滿分為100分)的{feeling}心得", "output: "])
        learning_area = model.generate_content(response.candidates[0].content.parts[0].text + "請把大學常用單字翻譯成英文, 若文本為英文則翻譯成中文, 並以第一行為*原文單字:翻譯單字*及第二行為*例句: 翻譯單字例句並換行2次格式") 
        return remove_mark(response.candidates[0].content.parts[0].text), mark, remove_mark(learning_area.candidates[0].content.parts[0].text)
    elif audio and course and mark != -1:
        audiocontent = audioai(audio)
        response = model.generate_content(traindata + ["input: " + audiocontent["text"] + f",請根據上述影片內容生成{course}課程的且符合該{mark}分(滿分為100分)的{type}心得", "output: "])
        learning_area = model.generate_content(response.candidates[0].content.parts[0].text + "請把大學常用單字翻譯成英文, 若文本為英文則翻譯成中文, 並以第一行為*原文單字:翻譯單字*及第二行為*例句: 翻譯單字例句並換行2次格式") 
        return remove_mark(response.candidates[0].content.parts[0].text), mark, remove_mark(learning_area.candidates[0].content.parts[0].text)
    elif audio and feeling and mark != -1:
        audiocontent = audioai(audio)
        response = model.generate_content(traindata + ["input: " + audiocontent["text"] + f",請根據上述影片內容生成課程的{feeling}且符合該{mark}分(滿分為100分)的{type}心得", "output: "])
        learning_area = model.generate_content(response.candidates[0].content.parts[0].text + "請把大學常用單字翻譯成英文, 若文本為英文則翻譯成中文, 並以第一行為*原文單字:翻譯單字*及第二行為*例句: 翻譯單字例句並換行2次格式") 
        return remove_mark(response.candidates[0].content.parts[0].text), mark, remove_mark(learning_area.candidates[0].content.parts[0].text)
    elif audio and wordlimit > 0 and mark != -1:
        audiocontent = audioai(audio)
        response = model.generate_content(traindata + ["input: " + audiocontent["text"] + f",請根據上述影片內容生成課程的且符合該{mark}分(滿分為100分)的{type}心得並限制在{wordlimit}字內", "output: "])
        learning_area = model.generate_content(response.candidates[0].content.parts[0].text + "請把大學常用單字翻譯成英文, 若文本為英文則翻譯成中文, 並以第一行為*原文單字:翻譯單字*及第二行為*例句: 翻譯單字例句並換行2次格式") 
        return remove_mark(response.candidates[0].content.parts[0].text), mark, remove_mark(learning_area.candidates[0].content.parts[0].text)
    elif audio and mark != -1:
        audiocontent = audioai(audio)
        response = model.generate_content(traindata + ["input: " + audiocontent["text"] + f",請根據上述影片內容生成課程的且符合該{mark}分(滿分為100分)的{type}心得", "output: "])
        learning_area = model.generate_content(response.candidates[0].content.parts[0].text + "請把大學常用單字翻譯成英文, 若文本為英文則翻譯成中文, 並以第一行為*原文單字:翻譯單字*及第二行為*例句: 翻譯單字例句並換行2次格式") 
        return remove_mark(response.candidates[0].content.parts[0].text), mark, remove_mark(learning_area.candidates[0].content.parts[0].text)
    
    elif audio and wordlimit > 0 and feeling and AImark:
        audiocontent = audioai(audio)
        response = model.generate_content(traindata + ["input: " + audiocontent["text"] + f",請根據上述影片內容生成課程的{feeling}心得並限制在{wordlimit}字內", "output: "])
        askmark = model.generate_content(response.candidates[0].content.parts[0].text + "請根據上文課程描述為課程評分, 只需回應1-100的數字")
        mark = askmark.candidates[0].content.parts[0].text
        learning_area = model.generate_content(response.candidates[0].content.parts[0].text + "請把大學常用單字翻譯成英文, 若文本為英文則翻譯成中文, 並以第一行為*原文單字:翻譯單字*及第二行為*例句: 翻譯單字例句並換行2次格式") 
        return remove_mark(response.candidates[0].content.parts[0].text), mark, remove_mark(learning_area.candidates[0].content.parts[0].text)
    elif audio and course and wordlimit > 0 and AImark:
        audiocontent = audioai(audio)
        response = model.generate_content(traindata + ["input: " + audiocontent["text"] + f",請根據上述影片內容生成{course}課程的心得並限制在{wordlimit}字內", "output: "])
        askmark = model.generate_content(response.candidates[0].content.parts[0].text + "請根據上文課程描述為課程評分, 只需回應1-100的數字")
        mark = askmark.candidates[0].content.parts[0].text
        learning_area = model.generate_content(response.candidates[0].content.parts[0].text + "請把大學常用單字翻譯成英文, 若文本為英文則翻譯成中文, 並以第一行為*原文單字:翻譯單字*及第二行為*例句: 翻譯單字例句並換行2次格式") 
        return remove_mark(response.candidates[0].content.parts[0].text), mark, remove_mark(learning_area.candidates[0].content.parts[0].text)
    elif audio and course and feeling and AImark:
        audiocontent = audioai(audio)
        response = model.generate_content(traindata + ["input: " + audiocontent["text"] + f",請根據上述影片內容生成{course}課程的{feeling}心得", "output: "])
        askmark = model.generate_content(response.candidates[0].content.parts[0].text + "請根據上文課程描述為課程評分, 只需回應1-100的數字")
        mark = askmark.candidates[0].content.parts[0].text
        learning_area = model.generate_content(response.candidates[0].content.parts[0].text + "請把大學常用單字翻譯成英文, 若文本為英文則翻譯成中文, 並以第一行為*原文單字:翻譯單字*及第二行為*例句: 翻譯單字例句並換行2次格式") 
        return remove_mark(response.candidates[0].content.parts[0].text), mark, remove_mark(learning_area.candidates[0].content.parts[0].text)
    elif audio and course and AImark:
        audiocontent = audioai(audio)
        response = model.generate_content(traindata + ["input: " + audiocontent["text"] + f",請根據上述影片內容生成{course}課程的心得", "output: "])
        askmark = model.generate_content(response.candidates[0].content.parts[0].text + "請根據上文課程描述為課程評分, 只需回應1-100的數字")
        mark = askmark.candidates[0].content.parts[0].text
        learning_area = model.generate_content(response.candidates[0].content.parts[0].text + "請把大學常用單字翻譯成英文, 若文本為英文則翻譯成中文, 並以第一行為*原文單字:翻譯單字*及第二行為*例句: 翻譯單字例句並換行2次格式") 
        return remove_mark(response.candidates[0].content.parts[0].text), mark, remove_mark(learning_area.candidates[0].content.parts[0].text)
    elif audio and feeling and AImark:
        audiocontent = audioai(audio)
        response = model.generate_content(traindata + ["input: " + audiocontent["text"] + f",請根據上述影片內容生成課程的{feeling}心得", "output: "])
        askmark = model.generate_content(response.candidates[0].content.parts[0].text + "請根據上文課程描述為課程評分, 只需回應1-100的數字")
        mark = askmark.candidates[0].content.parts[0].text
        learning_area = model.generate_content(response.candidates[0].content.parts[0].text + "請把大學常用單字翻譯成英文, 若文本為英文則翻譯成中文, 並以第一行為*原文單字:翻譯單字*及第二行為*例句: 翻譯單字例句並換行2次格式") 
        return remove_mark(response.candidates[0].content.parts[0].text), mark, remove_mark(learning_area.candidates[0].content.parts[0].text)
    elif audio and wordlimit > 0 and AImark:
        audiocontent = audioai(audio)
        response = model.generate_content(traindata + ["input: " + audiocontent["text"] + f",請根據上述影片內容生成課程的心得並限制在{wordlimit}字內", "output: "])
        askmark = model.generate_content(response.candidates[0].content.parts[0].text + "請根據上文課程描述為課程評分, 只需回應1-100的數字")
        mark = askmark.candidates[0].content.parts[0].text
        learning_area = model.generate_content(response.candidates[0].content.parts[0].text + "請把大學常用單字翻譯成英文, 若文本為英文則翻譯成中文, 並以第一行為*原文單字:翻譯單字*及第二行為*例句: 翻譯單字例句並換行2次格式") 
        return remove_mark(response.candidates[0].content.parts[0].text), mark, remove_mark(learning_area.candidates[0].content.parts[0].text)
    elif audio and AImark:
        audiocontent = audioai(audio)
        response = model.generate_content(traindata + ["input: " + audiocontent["text"] + f",請根據上述影片內容生成課程的心得字內", "output: "])
        askmark = model.generate_content(response.candidates[0].content.parts[0].text + "請根據上文課程描述為課程評分, 只需回應1-100的數字")
        mark = askmark.candidates[0].content.parts[0].text
        learning_area = model.generate_content(response.candidates[0].content.parts[0].text + "請把大學常用單字翻譯成英文, 若文本為英文則翻譯成中文, 並以第一行為*原文單字:翻譯單字*及第二行為*例句: 翻譯單字例句並換行2次格式") 
        return remove_mark(response.candidates[0].content.parts[0].text), mark, remove_mark(learning_area.candidates[0].content.parts[0].text)
    
    elif audio and wordlimit > 0 and feeling and course:
        audiocontent = audioai(audio)
        response = model.generate_content(traindata + ["input: " + audiocontent["text"] + f",請根據上述影片內容生成請生成{course}課程的{feeling}的心得並限制在{wordlimit}字內", "output: "])
        learning_area = model.generate_content(response.candidates[0].content.parts[0].text + "請把大學常用單字翻譯成英文, 若文本為英文則翻譯成中文, 並以第一行為*原文單字:翻譯單字*及第二行為*例句: 翻譯單字例句並換行2次格式") 
        return remove_mark(response.candidates[0].content.parts[0].text), -1, remove_mark(learning_area.candidates[0].content.parts[0].text)
    elif audio and wordlimit > 0 and feeling:
        audiocontent = audioai(audio)
        response = model.generate_content(traindata + ["input: " + audiocontent["text"] + f",請根據上述影片內容生成請生成課程的{feeling}心得並限制在{wordlimit}字內", "output: "])
        learning_area = model.generate_content(response.candidates[0].content.parts[0].text + "請把大學常用單字翻譯成英文, 若文本為英文則翻譯成中文, 並以第一行為*原文單字:翻譯單字*及第二行為*例句: 翻譯單字例句並換行2次格式") 
        return remove_mark(response.candidates[0].content.parts[0].text), -1, remove_mark(learning_area.candidates[0].content.parts[0].text)
    elif audio and wordlimit > 0 and course:
        audiocontent = audioai(audio)
        response = model.generate_content(traindata + ["input: " + audiocontent["text"] + f",請根據上述影片內容生成請生成{course}課程的心得並限制在{wordlimit}字內", "output: "])
        learning_area = model.generate_content(response.candidates[0].content.parts[0].text + "請把大學常用單字翻譯成英文, 若文本為英文則翻譯成中文, 並以第一行為*原文單字:翻譯單字*及第二行為*例句: 翻譯單字例句並換行2次格式") 
        return remove_mark(response.candidates[0].content.parts[0].text), -1, remove_mark(learning_area.candidates[0].content.parts[0].text)
    elif audio and feeling and course:
        audiocontent = audioai(audio)
        response = model.generate_content(traindata + ["input: " + audiocontent["text"] + f",請根據上述影片內容生成請生成{course}課程的{feeling}心得並限制在{wordlimit}字內", "output: "])
        learning_area = model.generate_content(response.candidates[0].content.parts[0].text + "請把大學常用單字翻譯成英文, 若文本為英文則翻譯成中文, 並以第一行為*原文單字:翻譯單字*及第二行為*例句: 翻譯單字例句並換行2次格式") 
        return remove_mark(response.candidates[0].content.parts[0].text), -1, remove_mark(learning_area.candidates[0].content.parts[0].text)
    elif audio and feeling:
        audiocontent = audioai(audio)
        response = model.generate_content(traindata + ["input: " + audiocontent["text"] + f",請根據上述影片內容生成請生成課程的{feeling}心得", "output: "])
        learning_area = model.generate_content(response.candidates[0].content.parts[0].text + "請把大學常用單字翻譯成英文, 若文本為英文則翻譯成中文, 並以第一行為*原文單字:翻譯單字*及第二行為*例句: 翻譯單字例句並換行2次格式") 
        return remove_mark(response.candidates[0].content.parts[0].text), -1, remove_mark(learning_area.candidates[0].content.parts[0].text)
    elif audio and wordlimit > 0:
        audiocontent = audioai(audio)
        response = model.generate_content(traindata + ["input: " + audiocontent["text"] + f",請根據上述影片內容生成請生成課程的心得並限制在{wordlimit}字內", "output: "])
        learning_area = model.generate_content(response.candidates[0].content.parts[0].text + "請把大學常用單字翻譯成英文, 若文本為英文則翻譯成中文, 並以第一行為*原文單字:翻譯單字*及第二行為*例句: 翻譯單字例句並換行2次格式") 
        return remove_mark(response.candidates[0].content.parts[0].text), -1, remove_mark(learning_area.candidates[0].content.parts[0].text)
    elif audio and course:
        audiocontent = audioai(audio)
        response = model.generate_content(traindata + ["input: " + audiocontent["text"] + f",請根據上述影片內容生成請生成{course}課程的心得", "output: "])
        learning_area = model.generate_content(response.candidates[0].content.parts[0].text + "請把大學常用單字翻譯成英文, 若文本為英文則翻譯成中文, 並以第一行為*原文單字:翻譯單字*及第二行為*例句: 翻譯單字例句並換行2次格式") 
        return remove_mark(response.candidates[0].content.parts[0].text), -1, remove_mark(learning_area.candidates[0].content.parts[0].text)
    
    elif audio:
        audiocontent = audioai(audio)
        response = model.generate_content(traindata + ["input: " + audiocontent["text"] + f",請根據上述影片內容生成請生成課程的心得", "output: "])
        learning_area = model.generate_content(response.candidates[0].content.parts[0].text + "請把大學常用單字翻譯成英文, 若文本為英文則翻譯成中文, 並以第一行為*原文單字:翻譯單字*及第二行為*例句: 翻譯單字例句並換行2次格式") 
        return remove_mark(response.candidates[0].content.parts[0].text), -1, remove_mark(learning_area.candidates[0].content.parts[0].text)
    
    # text part v1
    elif course and AImark and wordlimit > 0 and feeling:
        prompt = "input: 請生成{course}課程的{feeling}心得並限制在{wordlimit}字內"
        prompt = prompt.format(course=course, feeling=feeling, wordlimit=wordlimit)
        response = model.generate_content(traindata + [prompt, "output: "])
        askmark = model.generate_content(response.candidates[0].content.parts[0].text + "請根據上文課程描述為課程評分, 只需回應1-100的數字")
        mark = askmark.candidates[0].content.parts[0].text
        learning_area = model.generate_content(response.candidates[0].content.parts[0].text + "請把大學常用單字翻譯成英文, 若文本為英文則翻譯成中文, 並以第一行為*原文單字:翻譯單字*及第二行為*例句: 翻譯單字例句並換行2次格式") 
        return remove_mark(response.candidates[0].content.parts[0].text), mark, remove_mark(learning_area.candidates[0].content.parts[0].text)
    elif course and wordlimit > 0 and feeling and mark != -1:
        prompt = "input: 請生成{course}課程的{feeling}且符合該{mark}分(滿分為100分)的{type}心得並限制在{wordlimit}字內"
        prompt = prompt.format(course=course, feeling=feeling, mark=mark, type=type, wordlimit=wordlimit)
        response = model.generate_content(traindata + [prompt, "output: "])
        learning_area = model.generate_content(response.candidates[0].content.parts[0].text + "請把大學常用單字翻譯成英文, 若文本為英文則翻譯成中文, 並以第一行為*原文單字:翻譯單字*及第二行為*例句: 翻譯單字例句並換行2次格式") 
        return remove_mark(response.candidates[0].content.parts[0].text), mark, remove_mark(learning_area.candidates[0].content.parts[0].text)
    elif course and wordlimit > 0 and feeling:
        prompt = "input: 請生成{course}課程的{feeling}心得並限制在{wordlimit}字內"
        prompt = prompt.format(course=course, feeling=feeling, wordlimit=wordlimit)
        response = model.generate_content(traindata + [prompt, "output: "])
        learning_area = model.generate_content(response.candidates[0].content.parts[0].text + "請把大學常用單字翻譯成英文, 若文本為英文則翻譯成中文, 並以第一行為*原文單字:翻譯單字*及第二行為*例句: 翻譯單字例句並換行2次格式") 
        return remove_mark(response.candidates[0].content.parts[0].text), -1, remove_mark(learning_area.candidates[0].content.parts[0].text)
    elif course and AImark and wordlimit > 0:
        prompt = "input: 請生成{course}課程心得並限制在{wordlimit}字內"
        prompt = prompt.format(course=course, wordlimit=wordlimit)
        response = model.generate_content(traindata + [prompt, "output: "])
        askmark = model.generate_content(response.candidates[0].content.parts[0].text + "請根據上文課程描述為課程評分, 只需回應1-100的數字")
        mark = askmark.candidates[0].content.parts[0].text
        learning_area = model.generate_content(response.candidates[0].content.parts[0].text + "請把大學常用單字翻譯成英文, 若文本為英文則翻譯成中文, 並以第一行為*原文單字:翻譯單字*及第二行為*例句: 翻譯單字例句並換行2次格式") 
        return remove_mark(response.candidates[0].content.parts[0].text), mark, remove_mark(learning_area.candidates[0].content.parts[0].text)
    elif course and AImark and feeling:
        prompt = "input: 請生成{course}課程的{feeling}心得"
        prompt = prompt.format(course=course, feeling=feeling)
        response = model.generate_content(traindata + [prompt, "output: "])
        askmark = model.generate_content(response.candidates[0].content.parts[0].text + "請根據上文課程描述為課程評分, 只需回應1-100的數字")
        mark = askmark.candidates[0].content.parts[0].text
        learning_area = model.generate_content(response.candidates[0].content.parts[0].text + "請把大學常用單字翻譯成英文, 若文本為英文則翻譯成中文, 並以第一行為*原文單字:翻譯單字*及第二行為*例句: 翻譯單字例句並換行2次格式") 
        return remove_mark(response.candidates[0].content.parts[0].text), mark, remove_mark(learning_area.candidates[0].content.parts[0].text)
    elif course and mark != -1 and wordlimit > 0:
        prompt = "input: 請生成{course}課程且符合該{mark}分(滿分為100分)的{type}心得並限制在{wordlimit}字內"
        prompt = prompt.format(course=course, mark=mark, type=type, wordlimit=wordlimit)
        response = model.generate_content(traindata + [prompt, "output: "])
        learning_area = model.generate_content(response.candidates[0].content.parts[0].text + "請把大學常用單字翻譯成英文, 若文本為英文則翻譯成中文, 並以第一行為*原文單字:翻譯單字*及第二行為*例句: 翻譯單字例句並換行2次格式") 
        return remove_mark(response.candidates[0].content.parts[0].text), mark, remove_mark(learning_area.candidates[0].content.parts[0].text)
    elif course and mark != -1 and feeling:
        prompt = "input: 請生成{course}課程的{feeling}且符合該{mark}分(滿分為100分)的{type}心得"
        prompt = prompt.format(course=course, mark=mark, type=type, feeling=feeling)
        response = model.generate_content(traindata + [prompt, "output: "])
        learning_area = model.generate_content(response.candidates[0].content.parts[0].text + "請把大學常用單字翻譯成英文, 若文本為英文則翻譯成中文, 並以第一行為*原文單字:翻譯單字*及第二行為*例句: 翻譯單字例句並換行2次格式") 
        return remove_mark(response.candidates[0].content.parts[0].text), mark, remove_mark(learning_area.candidates[0].content.parts[0].text)
    elif course and mark != -1:
        prompt = "input: 請生成{course}課程且符合該{mark}分(滿分為100分)的{type}心得"
        prompt = prompt.format(course=course, mark=mark, type=type)
        response = model.generate_content(traindata + [prompt, "output: "])
        learning_area = model.generate_content(response.candidates[0].content.parts[0].text + "請把大學常用單字翻譯成英文, 若文本為英文則翻譯成中文, 並以第一行為*原文單字:翻譯單字*及第二行為*例句: 翻譯單字例句並換行2次格式") 
        return remove_mark(response.candidates[0].content.parts[0].text), mark, remove_mark(learning_area.candidates[0].content.parts[0].text)
    elif course and feeling:
        prompt = "input: 請生成{course}課程的{feeling}心得"
        prompt = prompt.format(course=course, feeling=feeling)
        response = model.generate_content(traindata + [prompt, "output: "])
        learning_area = model.generate_content(response.candidates[0].content.parts[0].text + "請把大學常用單字翻譯成英文, 若文本為英文則翻譯成中文, 並以第一行為*原文單字:翻譯單字*及第二行為*例句: 翻譯單字例句並換行2次格式") 
        return remove_mark(response.candidates[0].content.parts[0].text), -1, remove_mark(learning_area.candidates[0].content.parts[0].text)
    elif course and wordlimit > 0:
        prompt = "input: 請生成{course}課程心得並限制在{wordlimit}字內"
        prompt = prompt.format(course=course, wordlimit=wordlimit)
        response = model.generate_content(traindata + [prompt, "output: "])
        learning_area = model.generate_content(response.candidates[0].content.parts[0].text + "請把大學常用單字翻譯成英文, 若文本為英文則翻譯成中文, 並以第一行為*原文單字:翻譯單字*及第二行為*例句: 翻譯單字例句並換行2次格式") 
        return remove_mark(response.candidates[0].content.parts[0].text), -1, remove_mark(learning_area.candidates[0].content.parts[0].text)
    elif course and AImark:
        prompt = "input: 請生成{course}課程心得"
        prompt = prompt.format(course=course)
        response = model.generate_content(traindata + [prompt, "output: "])
        askmark = model.generate_content(response.candidates[0].content.parts[0].text + "請根據上文課程描述為課程評分, 只需回應1-100的數字")
        mark = askmark.candidates[0].content.parts[0].text
        learning_area = model.generate_content(response.candidates[0].content.parts[0].text + "請把大學常用單字翻譯成英文, 若文本為英文則翻譯成中文, 並以第一行為*原文單字:翻譯單字*及第二行為*例句: 翻譯單字例句並換行2次格式") 
        return remove_mark(response.candidates[0].content.parts[0].text), mark, remove_mark(learning_area.candidates[0].content.parts[0].text)
    elif course:
        prompt = "input: 請生成{course}課程心得"
        prompt = prompt.format(course=course)
        response = model.generate_content(traindata + [prompt, "output: "])
        learning_area = model.generate_content(response.candidates[0].content.parts[0].text + "請把大學常用單字翻譯成英文, 若文本為英文則翻譯成中文, 並以第一行為*原文單字:翻譯單字*及第二行為*例句: 翻譯單字例句並換行2次格式") 
        return remove_mark(response.candidates[0].content.parts[0].text), -1, remove_mark(learning_area.candidates[0].content.parts[0].text)
    return "", -1, ""

# 创建 Gradio 接口
demo = gr.Interface(
    fn=main,
    inputs=[gr.File(label="上傳mp3檔案"), "text", "number", "text", gr.Checkbox(label="AI mark"), gr.Slider(-1, 100)],
    outputs=[gr.Textbox(label="心得"), gr.Textbox(label="分數"), gr.Textbox(label="單字重點")],
    title="AI生成課程心得",
    description="若mark為-1即不需要評分",
)

# 启动 Gradio 应用程序
demo.launch()