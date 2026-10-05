import pandas as pd
from google import genai
import os
import time

# =================================================================
# 1. 配置 Gemini API Client
# =================================================================

# ⚠️ 這是您提供的 Key。請確認它是一個有效的 Gemini API Key。
# 務必保護您的金鑰，建議在部署時使用環境變數！
# API_KEY = 'AIzaSyBvVsAlSAeShJO80dgRPZvpHcorcLPhskM' # cindyhaung5741@gmail.com
API_KEY = 'AIzaSyBvVsAlSAeShJO80dgRPZvpHcorcLPhskM' #yanshinhuang0830

try:
    # 🚀 修正：直接使用 API_KEY 變數進行初始化。
    # 這樣可以確保即使環境變數未設定，也能成功連線。
    client = genai.Client(api_key=API_KEY)
    print("✅ Client 已使用程式碼內的 API Key 變數成功初始化。")
except Exception as e:
    # 這裡捕獲更通用的錯誤，如果 API Key 本身無效或連線有問題
    print(f"❌ 無法初始化 Gemini Client (請檢查 API Key 是否有效): {e}")
    # 將 client 設為 None，讓後續處理邏輯知道無法執行 AI 呼叫
    client = None


def call_gemini_for_question(context_text):
    """
    使用 Gemini API 根據上下文生成一個擬真問句。
    :param context_text: 包含上下文內容。
    :return: AI 生成的擬真問句 (字串) 或錯誤訊息。
    """
    if client is None:
        return "AI_GENERATION_FAILED: Client 初始化失敗"
        
    # 這是您提供的專業 Prompt
    prompt = (
        f"You are a **data augmentation expert** specializing in generating customer service paraphrased questions.\n\n"
        f"# Context\n"
        f"This dataset is for a banking chatbot. The data provided below (Original Question and Intent) is used as context.\n\n"
        f"# Input Data\n"
        f"{context_text}\n\n"
        f"# Task\n"
        f"Generate about **1 natural, fluent, realistic paraphrased customer question** that expresses the **same intent**, written in **a different tone or customer persona**.\n\n"
        f"✅ Requirements:\n"
        f"- Do not copy or repeat any of the input data (standard question or text).\n"
        f"- Output only one complete question.\n"
        f"- Keep the question realistic and aligned with the original intent.\n"
        f"- Write in a different customer tone or style (e.g., impatient, polite, young, elderly, concise, confused, suspicious, foreign, professional, friendly).\n"
        f"- Do not include any emojis or emotional words.\n"
        f"- Only output the question, nothing else, no numbering, comments, or explanations."
    )

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash', # 輕量且快速的模型
            contents=prompt
        )
        
        # 返回 AI 生成的問句，並清除多餘的空白和換行符
        return response.text.strip().replace('\n', ' ').replace('\r', '') 
    
    except Exception as e:
        # 如果呼叫 API 失敗，返回錯誤標記
        print(f"❌ Gemini API 呼叫失敗: {e}")
        return f"AI_GENERATION_FAILED_AT_{time.strftime('%H:%M:%S')}"


# =================================================================
# 2. Excel 批次處理邏輯 (使用 Pandas)
# =================================================================

def process_excel_with_ai(input_file, output_file, column_to_fill='擬真問句'):
    """
    讀取 Excel 檔案，針對指定欄位中的空格，使用 AI 生成內容來填補。
    """
    if client is None:
        print("❌ 無法執行批次處理，因為 Gemini Client 初始化失敗。")
        return
        
    try:
        # 讀取 .xlsx 檔案，使用 read_excel
        # 假設資料在第一個工作表 (sheet_name=0)
        df = pd.read_excel(input_file, sheet_name=0) 
        print(f"✅ 成功讀取檔案: {input_file}。總共 {len(df)} 筆資料。")
        
        # 找出需要填補的列（在目標欄位中是空值 NaN）
        blanks_index = df[df[column_to_fill].isna()].index
        print(f"🔍 找到 {len(blanks_index)} 處空格需要 AI 填補...")
        
        if len(blanks_index) == 0:
            print("✨ 檔案中沒有需要填補的空格，無需處理。")
            return

        # 定義哪些欄位將作為上下文提供給 AI 
        CONTEXT_COLUMNS = ['編號', '標準問句', 'text']
        
        for count, index in enumerate(blanks_index):
            print(f"\n--- 正在處理第 {count + 1} 處空格 (總計 {len(blanks_index)}) ---")
            
            # 取得該列特定欄位的所有非空資料作為 AI 上下文
            context_data = df.loc[index, CONTEXT_COLUMNS].dropna().astype(str)
            
            # 將非空的資料組合成一個提示字串
            context = "\n".join([f"- {k}: {v}" for k, v in context_data.items()])
            
            if context_data.empty:
                df.loc[index, column_to_fill] = "無足夠上下文資料 (Not Enough Context)"
                print("  > 該列無足夠資訊，跳過 AI 處理。")
            else:
                print(f"  > 呼叫 AI (輸入內容):\n{context}")
                
                # 呼叫 AI 機器人生成擬真問句
                ai_generated_question = call_gemini_for_question(context)
                
                # 將生成的問句填回 DataFrame
                df.loc[index, column_to_fill] = ai_generated_question
                print(f"  > 填入問句: {ai_generated_question}")
                
            # 每次呼叫後暫停 1 秒，避免速率限制
            time.sleep(2)
            
        # 儲存為 .xlsx 檔案，使用 to_excel
        df.to_excel(output_file, index=False)
        print(f"\n==================================================")
        print(f"💾 所有處理完成。結果已儲存至新檔案: {output_file}")
        print(f"==================================================")
        
    except FileNotFoundError:
        print(f"❌ 錯誤: 找不到檔案 {input_file}。請確認檔案名稱和路徑是否正確。")
    except KeyError:
        # 如果欄位名稱錯誤 (例如 '擬真問句' 寫錯了)
        print(f"❌ 錯誤: 檔案中找不到名為 '{column_to_fill}' 的欄位。請檢查 Excel 標題名稱是否正確。")
    except Exception as e:
        print(f"❌ 發生未預期的錯誤: {e}")

# =================================================================
# 3. 執行程式
# =================================================================

# ⚠️ 請確保這兩個檔案位於您的 Python 腳本的同一目錄下，或使用完整的檔案路徑
input_file = '10萬句llm生成擬真問句_去重.xlsx'
output_file = '給gemini跑_AI填補完成10萬句llm生成擬真問句.xlsx'

# 填寫目標欄位名稱 
target_column = '擬真問句' 

# 執行批次處理
process_excel_with_ai(input_file, output_file, target_column)