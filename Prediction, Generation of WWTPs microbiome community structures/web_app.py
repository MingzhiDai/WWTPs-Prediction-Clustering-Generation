import streamlit as st
import subprocess
import os
import numpy as np
import time
import glob
import pandas as pd

st.set_page_config(page_title="DE-BP Neural Network Optimization Mini-Program", layout="centered")
st.title("🧠 DE-BP Neural Network Genetic Algorithm Optimization")

# ✅ 初始化 session_state（仅执行一次）
if "Run_finished" not in st.session_state:
    st.session_state["Run_finished"] = False
if "output_dir" not in st.session_state:
    st.session_state["output_dir"] = ""

# --- 参数输入区 ---
st.header("参数输入 Parameter Input")

filename = st.text_input("输入文件名（含路径）Input filename (include file path)", "D:/BaiduNetdiskDownload/Sample_information_final-Phylums.csv")
n_feature = st.number_input("神经网络输入节点数 Input input_node_number: M [Default 37]", min_value=1, value=37, step=1)
n_hidden = st.number_input("神经网络隐层节点数  Input hidden_node_number: √(M + N) + α α∈[1, 10]", min_value=1, value=15, step=1)
n_output = st.number_input("神经网络输出节点数  Input output_node_number: N [Phylums 21; Class 51; Order 171]", min_value=1, value=21, step=1)
num_epoch = st.number_input("神经网络迭代次数 DE-BP Iteration Count"  , min_value=1, value=1500, step=1)
learn_rate = st.number_input("学习率 learning rate", min_value=0.0001, max_value=1.0, value=0.01, step=0.001, format="%.4f")
population_size = st.number_input("DE 种群规模 DE population scale", min_value=1, value=15, step=1)
p_cross = st.slider("交叉概率 crossover rate", 0.0, 1.0, 0.4)
p_mutate = st.slider("变异概率 mutation rate", 0.0, 0.1, 0.01)
maxgen = st.number_input("DE 最大迭代次数 DE maximum iteration", min_value=1, value=20, step=1)
output_dir = st.text_input("输出文件夹（路径）Output filename (include file path)", "D:/BaiduNetdiskDownload/Data/Phylum")
test_instance = st.number_input("测试实例组序号 Test instance group number: M", min_value=1, value=1, step=1)

st.markdown("------")

# --- 运行按钮 ---
if st.button("▶️ Run DE-BP model"):
    st.info("Running，please wait...")

    # 初始化进度条
    progress_bar = st.progress(0)
    progress_text = st.empty()

    total_steps = maxgen
    for i in range(total_steps):
        progress = int((i + 1) / total_steps * 100)
        progress_bar.progress(progress)
        progress_text.text(f"Program Status：{progress}%")
        time.sleep(0.2)  # 模拟每代计算时间

    try:
        cmd = [
            "python", "gabp.py",
            "--file", filename,
            "--n_feature", str(n_feature),
            "--n_hidden", str(n_hidden),
            "--n_output", str(n_output),
            "--num_epoch", str(num_epoch),
            "--learn_rate", str(learn_rate),
            "--population_size", str(population_size),
            "--p_cross", str(p_cross),
            "--p_mutate", str(p_mutate),
            "--maxgen", str(maxgen),
            "--output_dir", output_dir,
            "--test_instance", str(test_instance)
        ]

        subprocess.run(cmd, check=True)

        progress_bar.progress(100)
        progress_text.text("✅ Run Complete! ")
        st.success("Program execution completed! ✅")

        # ✅ 保存状态
        st.session_state["Run_finished"] = True
        st.session_state["output_dir"] = output_dir

    except subprocess.CalledProcessError as e:
        st.error(f"Run failed：{e}")
        st.session_state["Run finished"] = False


# --- ✅ 展示结果区 ---
if st.session_state["Run_finished"]:
    st.subheader("📄📥 All Results Display and Download")

    output_dir = st.session_state["output_dir"]
    output_path = os.path.join(output_dir, "Result.txt")
    log_path = os.path.join(output_dir, "Gabp_log.xlsx")
    csv_path = os.path.join(output_dir, "Test_instance_output.csv")

    # --- Result.txt ---
    if os.path.exists(output_path):
        st.write("📄 Result.txt Preview：")
        with open(output_path, "r", encoding="utf-8") as f:
            st.text(f.read()[:1000])
        with open(output_path, "rb") as f:
            st.download_button("📥 Download Result.txt", f, file_name="Result.txt", key="dl_txt")
    else:
        st.warning("❌ Can‘t find Result.txt file.")

    # --- Gabp_log.xlsx ---
    if os.path.exists(log_path):
        st.write("📊 Gabp_log.xlsx Preview：")
        try:
            df_log = pd.read_excel(log_path)
            st.dataframe(df_log.head(10))
        except Exception as e:
            st.error(f"Can‘t Preview Excel file：{e}")
        with open(log_path, "rb") as f:
            st.download_button("📥 Download Gabp_log.xlsx", f, file_name="Gabp_log.xlsx", key="dl_xlsx")
    else:
        st.warning("❌ Can‘t find Gabp_log.xlsx file.")

    # --- Test_instance_output.csv ---
    if os.path.exists(csv_path):
        st.write("📊 Test_instance_output.csv Preview：")
        try:
            df_csv = pd.read_csv(csv_path, encoding="utf-8")
            st.dataframe(df_csv.head(10))  # 显示前 10 行
        except Exception as e:
            st.error(f"Can‘t find CSV file：{e}")

        # ✅ 下载按钮（使用不同 key 防止 Streamlit 冲突）
        with open(csv_path, "rb") as f:
            st.download_button(
                "📥 Download Test_instance_output.csv",
                f,
                file_name="Test_instance_output.csv",
                key="dl_csv"
            )
    else:
        st.warning("❌ Can‘t find Test_instance_output.csv file.")

    # --- 图像展示 ---
    st.subheader("🖼️ Model visualization results")

    pattern_list = [
        os.path.join(output_dir, "BP_prediction_*.png"),
        os.path.join(output_dir, "BP_error_drop_curve.png"),
        os.path.join(output_dir, "DE-BP_error_drop_curve.png"),
        os.path.join(output_dir, "DE-BP_prediction_*.png")
    ]

    for idx, pattern in enumerate(pattern_list):
        matched_files = sorted(glob.glob(pattern))
        if matched_files:
            st.markdown(f"**{os.path.basename(pattern)} Matching results：**")
            for i, file in enumerate(matched_files[:3]):
                st.image(file, caption=os.path.basename(file), use_container_width=True)
                with open(file, "rb") as f:
                    st.download_button(
                        label=f"📥 Download {os.path.basename(file)}",
                        data=f,
                        file_name=os.path.basename(file),
                        key=f"dl_img_{idx}_{i}"  # ✅ 每个按钮唯一 key
                    )
        else:
            st.info(f"Can‘t find {os.path.basename(pattern)} image files.")
else:
    st.info("👆 Click the button above to run gabp.py and view the results.")
