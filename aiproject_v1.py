import gradio as gr
import google.generativeai as genai
from getpass import getpass

# 获取并配置 Google API Key
GOOGLE_API_KEY = getpass(prompt="Enter your Google API Key: ")
genai.configure(api_key=GOOGLE_API_KEY)

# 创建生成模型实例
model = genai.GenerativeModel('gemini-1.5-pro')

# 定义主函数
def main(course, wordlimit, feeling, aimark, mark):
    if mark > 50:
        type = "正面"
    else:
        type = "負面"
    if course and aimark and wordlimit > 0 and feeling:
        response = model.generate_content(f"請生成{course}課程的{feeling}心得並限制在{wordlimit}字內")
        askmark = model.generate_content(response.candidates[0].content.parts[0].text + "請根據上文課程描述為課程評分, 只需回應1-100的數字")
        mark = askmark.candidates[0].content.parts[0].text
        return response.candidates[0].content.parts[0].text, mark
    elif course and wordlimit > 0 and feeling and mark != -1:
        response = model.generate_content(f"請生成{course}課程的{feeling}且符合該{mark}分(滿分為100分)的{type}心得並限制在{wordlimit}字內")
        return response.candidates[0].content.parts[0].text, mark
    elif course and wordlimit > 0 and feeling:
        response = model.generate_content(f"請生成{course}課程的{feeling}心得並限制在{wordlimit}字內")
        return response.candidates[0].content.parts[0].text, -999
    elif course and aimark and wordlimit > 0:
        response = model.generate_content(f"請生成{course}課程心得並限制在{wordlimit}字內")
        askmark = model.generate_content(response.candidates[0].content.parts[0].text + "請根據上文課程描述為課程評分, 只需回應1-100的數字")
        mark = askmark.candidates[0].content.parts[0].text
        return response.candidates[0].content.parts[0].text, mark
    elif course and aimark and feeling:
        response = model.generate_content(f"請生成{course}課程的{feeling}心得")
        askmark = model.generate_content(response.candidates[0].content.parts[0].text + "請根據上文課程描述為課程評分, 只需回應1-100的數字")
        mark = askmark.candidates[0].content.parts[0].text
        return response.candidates[0].content.parts[0].text, mark
    elif course and mark != -1 and wordlimit > 0:
        response = model.generate_content(f"請生成{course}課程且符合該{mark}分(滿分為100分)的{type}心得並限制在{wordlimit}字內")
        return response.candidates[0].content.parts[0].text, mark
    elif course and mark != -1 and feeling:
        response = model.generate_content(f"請生成{course}課程的{feeling}且符合該{mark}分(滿分為100分)的{type}心得")
        return response.candidates[0].content.parts[0].text, mark
    elif course and mark != -1:
        response = model.generate_content(f"請生成{course}課程且符合該{mark}分(滿分為100分)的{type}心得")
        return response.candidates[0].content.parts[0].text, mark
    elif course and feeling:
        response = model.generate_content(f"請生成{course}課程的{feeling}心得")
        return response.candidates[0].content.parts[0].text, -999
    elif course and wordlimit > 0:
        response = model.generate_content(f"請生成{course}課程心得並限制在{wordlimit}字內")
        return response.candidates[0].content.parts[0].text, -999
    elif course and aimark:
        response = model.generate_content(f"請生成{course}課程心得")
        askmark = model.generate_content(response.candidates[0].content.parts[0].text + "請根據上文課程描述為課程評分, 只需回應1-100的數字")
        mark = askmark.candidates[0].content.parts[0].text
        return response.candidates[0].content.parts[0].text, mark
    elif course:
        response = model.generate_content(f"請生成{course}課程心得")
        return response.candidates[0].content.parts[0].text, -999
    return "", -999

# 创建 Gradio 接口
demo = gr.Interface(
    fn=main,
    inputs=["text", "number", "text", "checkbox", gr.Slider(-1, 100)],
    outputs=[gr.Textbox(label="心得"), gr.Textbox(label="分數")],
    title="AI生成課程心得",
    description="若mark為-1即不需要評分",
)

# 启动 Gradio 应用程序
demo.launch()