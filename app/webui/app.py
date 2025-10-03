import streamlit as st
from openai_env import OPENAI_MODEL, SD_API, client
from openai_function import TOOLS , SYSTEM_PROMPT
from tools import  img_to_data_uri, call_sd_txt2img, call_sd_img2img
import json, io, base64
import requests


# ========= Streamlit UI =========
st.set_page_config(page_title="SD + LoRA + OpenAI Orchestrator", layout="centered")
st.title("🎨 SD(+LoRA) × OpenAI LLM Orchestrator")
st.caption("輸入文字，或同時上傳參考圖。LLM 自動選 txt2img / img2img，並下發正確參數。")

with st.form("gen"):
    user_text = st.text_area("你的需求（prompt 指令或自然語言都可）", height=120)
    uploaded = st.file_uploader("可選：上傳一張圖片（img2img 會使用）", type=["png", "jpg", "jpeg", "webp"])
    col1, col2 = st.columns(2)
    with col1:
        temperature = 1 # st.slider("LLM 溫度（越低越保守）", 0.0, 1.0, 0.2, 0.05)
    with col2:
        dry_run = st.checkbox("只看 LLM 產生的 JSON（不真的生圖）", value=False)
    submitted = st.form_submit_button("送出")

if submitted:
    if not user_text and not uploaded:
        st.warning("請至少輸入文字或上傳圖片。")
        st.stop()

    # 準備 user 訊息（含圖像 Data URL，供 LLM 理解）
    uploaded_b64 = None
    user_content = []
    if user_text:
        user_content.append({"type": "text", "text": user_text})
    if uploaded:
        uploaded_b64 = img_to_data_uri(uploaded)
        # 讓 LLM 看到圖片（用 vision）
        user_content.append({"type": "image_url", "image_url": {"url": uploaded_b64}})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": user_content
        },
        # 額外提示 LLM：目前是否有上傳圖（幫助路由判斷）
        {"role": "system", "content": f"使用者目前{'有' if uploaded_b64 else '沒有'}上傳圖片。"}
    ]

    # 第一次：請 LLM 決定是否要呼叫工具與填參數
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=temperature,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )

    msg = resp.choices[0].message
    tool_calls = msg.tool_calls or []

    if not tool_calls:
        st.error("模型沒有呼叫工具。可能是輸入不足或規則限制。")
        st.json({"llm_reply": msg})
        st.stop()

    # 目前設計：只處理第一個工具呼叫（多工具也能擴充成迴圈）
    tc = tool_calls[0]
    fn_name = tc.function.name
    fn_args = json.loads(tc.function.arguments or "{}")

    st.subheader("🧠 LLM 決策（工具與參數）")
    st.code(json.dumps({"tool": fn_name, "args": fn_args}, ensure_ascii=False, indent=2), language="json")

    if dry_run:
        st.info("Dry run：未呼叫 SD 後端。")
        st.stop()

    try:
        if fn_name == "txt2img":
            result = call_sd_txt2img(fn_args)
        elif fn_name == "img2img":
            result = call_sd_img2img(fn_args, uploaded_b64)
        else:
            raise ValueError(f"未知工具：{fn_name}")
    except requests.HTTPError as e:
        st.error(f"SD 後端錯誤：{e}\n{e.response.text if e.response is not None else ''}")
        st.stop()
    except Exception as e:
        st.error(f"工具執行失敗：{e}")
        st.stop()

    # 顯示生成圖片
    img_b64 = result["image_base64"].split(",", 1)[1]
    st.image(io.BytesIO(base64.b64decode(img_b64)), caption=f"seed={result['seed']}, size={result['width']}x{result['height']}")
    with st.expander("生成的中繼資訊"):
        st.json(result, expanded=False)

    # （可選）第二次：把工具輸出回饋給 LLM，產生一句自然語言說明
    follow_messages = messages + [
        msg,  # 原本模型訊息（含 tool_call）
        {
            "role": "tool",
            "tool_call_id": tc.id,
            "name": fn_name,
            "content": json.dumps({k: result[k] for k in ("seed", "width", "height", "steps", "guidance_scale", "applied_loras")}, ensure_ascii=False),
        },
        {"role": "user", "content": "請用一句話幫我總結這次的生成重點。"}
    ]
    explain = client.chat.completions.create(
        model=OPENAI_MODEL, temperature=1, messages=follow_messages
    )
    st.caption("📝 " + (explain.choices[0].message.content or "").strip())
