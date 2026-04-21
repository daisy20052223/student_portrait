import os
import streamlit as st
import pandas as pd
import numpy as np
from streamlit_echarts import st_echarts

st.set_page_config(page_title="学生群体画像", layout="wide")

st.title("📊 学生群体画像分析")

# =======================
# 0️⃣ 云端部署：自动查找文件路径
# =======================
def find_file(filename):
    """自动查找文件，适配本地和云端"""
    # 方法1：当前目录
    if os.path.exists(filename):
        return filename
    # 方法2：当前目录下的 data 文件夹
    if os.path.exists(os.path.join("data", filename)):
        return os.path.join("data", filename)
    # 方法3：上级目录
    if os.path.exists(os.path.join("..", filename)):
        return os.path.join("..", filename)
    # 方法4：使用 st.secrets 中的路径（云端专用）
    try:
        if filename in st.secrets.get("file_paths", {}):
            return st.secrets["file_paths"][filename]
    except:
        pass
    return filename

# =======================
# 1️⃣ 读取数据
# =======================
df = pd.read_csv(find_file("Result4_Individual_Diagnostic_Reports.csv"))
zscore_df = pd.read_csv(find_file("Result1_Cluster_Portrait_ZScore.csv"))
importance_df = pd.read_csv(find_file("Result2_Global_Feature_Importance.csv"))
individual_df = pd.read_csv(find_file("最终版个人诊断报告_风险解读版.csv"))
risk_df = pd.read_csv(find_file("A_Student_Risk_Scores_Final.csv"))
attribution_df = pd.read_csv(find_file("A_Final_Attribution_Report.csv"))
radar_df = pd.read_csv(find_file("Result3_Individual_Radar_Full_Data.csv"), index_col=0)
radar_df.index.name = 'student_id'

# 删除 cluster 列
if 'cluster' in radar_df.columns:
    radar_df = radar_df.drop('cluster', axis=1)
# 设置 cluster 为索引
zscore_df = zscore_df.set_index("cluster")

counts = df["cluster"].value_counts().sort_index()
total_students = counts.sum()

# =======================
# 2️⃣ 全局特征重要性数据处理
# =======================
# 使用 Final_Score 作为综合重要性指标
importance_df = importance_df.sort_values('Final_Score', ascending=False).reset_index(drop=True)

# 特征中文名映射
feature_names = {
    "学科竞赛_vector_composite.csv": "学科竞赛",
    "Scholarship_vector.csv": "奖学金获奖",
    "线上学习_vector_composite.csv": "线上学习",
    "Evaluation_vector.csv": "本科生综合测评",
    "成绩_vector_variable.csv": "学业成绩",
    "run_vector_data.xlsx": "跑步打卡",
    "Graduate_vector.csv": "毕业去向",
    "Portrait_vector.csv": "学生画像",
    "Tags_vector.csv": "上网行为标签",
    "atten_vector_data.xlsx": "考勤汇总",
    "dorm_vector_data.xlsx": "门禁数据",
    "student_behavior_vector.xlsx": "学生行为",
    "体测数据_特征轨迹_vector.xlsx": "体测数据",
    "体育课_特征向量_vector.xlsx": "体育课表现",
    "体能_vector_variable.csv": "体能测试",
    "作业提交_vector_composite.csv": "作业提交",
    "四六级_特征向量_vector.xlsx": "四六级成绩",
    "图书馆_特征轨迹_vector.xlsx": "图书馆利用",
    "基本信息_vector_variable.csv": "基本信息",
    "学生签到_特征向量.csv": "学生签到",
    "学籍异动_vector_composite.csv": "学籍异动",
    "日常锻炼_vector.xlsx": "日常锻炼",
    "社团活动_特征向量_vector.xlsx": "社团活动",
    "讨论记录_特征向量_vector.xlsx": "讨论记录",
    "选课_vector_variable.csv": "选课行为"
}

# 维度分类映射
dimension_map = {
    "学科竞赛": "学术硬核维度",
    "奖学金获奖": "学术硬核维度",
    "线上学习": "学术硬核维度",
    "本科生综合测评": "学术硬核维度",
    "学业成绩": "学术硬核维度",
    "四六级成绩": "学术硬核维度",
    "作业提交": "学术硬核维度",
    "考勤汇总": "生活自律维度",
    "门禁数据": "生活自律维度",
    "跑步打卡": "生活自律维度",
    "日常锻炼": "生活自律维度",
    "体测数据": "生活自律维度",
    "学生签到": "生活自律维度",
    "毕业去向": "志趣导向维度",
    "上网行为标签": "志趣导向维度",
    "社团活动": "志趣导向维度",
    "选课行为": "志趣导向维度",
    "图书馆利用": "志趣导向维度",
    "讨论记录": "志趣导向维度"
}

# 添加中文名和维度
importance_df['feature_cn'] = importance_df['Feature_Name'].map(lambda x: feature_names.get(x, x))
importance_df['dimension'] = importance_df['feature_cn'].map(lambda x: dimension_map.get(x, "其他维度"))

# 按维度分组统计（使用 Final_Score）
dimension_importance = importance_df.groupby('dimension')['Final_Score'].sum().sort_values(ascending=False)

# =======================
# 3️⃣ 映射名称 + 详细画像
# =======================
cluster_profiles = {
    0: {
        "name": "自律型领航者",
        "en_name": "Self-Regulated Achievers",
        "desc": "该群体在校园行为大数据中表现出极强的「学术极化」特征。从标准偏差来看，他们在学科竞赛与线上学习资源利用上呈现出显著的正向偏离，这反映了他们不仅在传统学业评价中处于顶尖地位，更具备主动寻求学术挑战和利用数字化资源的能力。简单来说，他们就是学校里的「全能领跑员」，不仅在硬核竞赛上拿奖拿到手软，平时的自主学习劲头也非常足，是校园里资源利用率最高、最具榜样作用的精英族群。",
        "summary": "🎯 全能领跑员 · 竞赛能手 · 资源利用王者",
        "color": "#4CAF50",
        "icon": "🏆"
    },
    1: {
        "name": "平衡型稳健生",
        "en_name": "Balanced Mainstreamers",
        "desc": "这一类学生构成了校园生态的「基本盘」，其各项行为特征高度趋近于全校正态分布的均值中心，Z-Score 波动区间极窄（均在 ±0.2 之间）。他们在学术、社交和日常行为逻辑上表现出极高的稳定性与合规性。他们是校园里那群最踏实的「中间力量」，虽然没有极端的表现，但胜在各方面均衡、情绪稳定且行为风险极低，是学校管理中最让人放心的「稳健派」。",
        "summary": "⚖️ 校园中坚 · 均衡稳定 · 低风险群体",
        "color": "#2196F3",
        "icon": "📚"
    },
    2: {
        "name": "非传统学习型",
        "en_name": "Non-traditionalists",
        "desc": "该簇群的行为画像展示了一种明显的「精力转移」倾向。他们在社团活动和选课活跃度上有轻微的正向表现，但学术产出的核心指标——如学科竞赛（-1.13）与奖学金获取（-1.08）——却出现了明显的负向偏移。这说明他们将更多的个人能量投入到了非学术性的素质拓展与社交互动中。这群同学在校园活动里非常活跃，人际关系可能处理得很溜，但在硬核学习和学术竞争上确实稍微「佛系」了一点，导致学术标签不够鲜明。",
        "summary": "🎭 社交活跃 · 学术佛系 · 精力转移",
        "color": "#FF9800",
        "icon": "🎪"
    },
    3: {
        "name": "学业游离型",
        "en_name": "Academic Disengagement",
        "desc": "这个群体在模型中触发了多维度的负向预警信号，表现出明显的「学业游离」态势。其核心成绩指标（-1.24）与四六级过级能力（-1.05）均大幅跌破基准线，且在线上学习参与度上存在巨大的断层（-1.17）。更值得关注的是，他们的上网行为和学籍异动风险已开始抬头。直白地说，这部分同学正处于学业「挂科」甚至退学的危险边缘，不仅平时学习跟不上节奏，甚至在行为轨迹上也逐渐脱离了校园正常轨道，是学校必须第一时间拉响预警、精准干预的重点对象。",
        "summary": "⚠️ 学业预警 · 边缘风险 · 急需干预",
        "color": "#F44336",
        "icon": "🚨"
    }
}

# 选择要展示的关键维度
key_dimensions = [
    "成绩_vector_variable.csv",
    "学科竞赛_vector_composite.csv",
    "线上学习_vector_composite.csv",
    "社团活动_特征向量_vector.xlsx",
    "四六级_特征向量_vector.xlsx",
    "图书馆_特征轨迹_vector.xlsx",
    "作业提交_vector_composite.csv",
    "学生签到_特征向量.csv"
]

# 维度中文名映射
dimension_names = {
    "成绩_vector_variable.csv": "学业成绩",
    "学科竞赛_vector_composite.csv": "学科竞赛",
    "线上学习_vector_composite.csv": "线上学习",
    "社团活动_特征向量_vector.xlsx": "社团活动",
    "四六级_特征向量_vector.xlsx": "英语能力",
    "图书馆_特征轨迹_vector.xlsx": "图书馆利用",
    "作业提交_vector_composite.csv": "作业提交",
    "学生签到_特征向量.csv": "出勤率"
}

# 只保留存在的维度
available_dims = [d for d in key_dimensions if d in zscore_df.columns]
available_dim_names = [dimension_names.get(d, d) for d in available_dims]


# =======================
# 4️⃣ 辅助函数
# =======================
def create_bar_chart(cluster_id):
    """为指定群体创建条形图"""
    if cluster_id not in zscore_df.index:
        return None

    zscore_values = []
    for dim in available_dims:
        val = zscore_df.loc[cluster_id, dim]
        zscore_values.append(round(float(val), 2))

    bar_colors = ["#4CAF50" if v >= 0 else "#F44336" for v in zscore_values]

    bar_option = {
        "title": {"show": False},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}, "formatter": "{b}<br/>Z-Score: {c}σ"},
        "grid": {"left": "15%", "right": "8%", "top": "5%", "bottom": "5%", "containLabel": True},
        "xAxis": {"type": "value", "name": "Z-Score", "axisLabel": {"formatter": "{value}σ"}},
        "yAxis": {"type": "category", "data": available_dim_names, "axisLabel": {"fontSize": 11}},
        "series": [{
            "name": "Z-Score", "type": "bar",
            "data": [{"value": v, "itemStyle": {"color": bar_colors[idx]}} for idx, v in enumerate(zscore_values)],
            "label": {"show": True, "position": "right", "formatter": "{c}σ", "fontSize": 10},
            "barWidth": "50%"
        }]
    }
    return bar_option


def create_feature_importance_chart():
    """创建全局特征重要性条形图（使用 pyecharts 风格简化配置）"""
    top_features = importance_df.head(10)

    # 准备数据 - 反转让最大值在顶部
    feature_names_list = top_features['feature_cn'].tolist()[::-1]
    scores = top_features['Final_Score'].tolist()[::-1]

    # 生成颜色渐变
    colors = ['#667eea', '#764ba2']

    option = {
        "tooltip": {
            "trigger": "axis",
            "axisPointer": {"type": "shadow"},
            "formatter": "{b}<br/>综合得分: {c}"
        },
        "grid": {
            "left": "3%",
            "right": "4%",
            "bottom": "3%",
            "top": "3%",
            "containLabel": True
        },
        "xAxis": {
            "type": "value",
            "name": "综合得分",
            "nameLocation": "middle",
            "nameGap": 30,
            "axisLabel": {"fontSize": 11}
        },
        "yAxis": {
            "type": "category",
            "data": feature_names_list,
            "axisLabel": {"fontSize": 12, "fontWeight": "bold"},
            "axisLine": {"show": False},
            "axisTick": {"show": False}
        },
        "series": [
            {
                "name": "综合得分",
                "type": "bar",
                "data": scores,
                "itemStyle": {
                    "borderRadius": [0, 4, 4, 0],
                    "color": {
                        "type": "linear",
                        "x": 0, "y": 0, "x2": 1, "y2": 0,
                        "colorStops": [
                            {"offset": 0, "color": "#667eea"},
                            {"offset": 1, "color": "#764ba2"}
                        ]
                    }
                },
                "label": {
                    "show": True,
                    "position": "right",
                    "formatter": "{c}",
                    "fontSize": 11,
                    "fontWeight": "bold"
                },
                "barWidth": "50%"
            }
        ]
    }
    return option


def create_dimension_pie_chart():
    """创建维度贡献占比饼图"""
    # 准备数据
    dimension_data = []
    for dim, imp in dimension_importance.items():
        dimension_data.append({"name": dim, "value": round(imp, 4)})

    option = {
        "tooltip": {
            "trigger": "item",
            "formatter": "{b}<br/>综合贡献: {c}<br/>占比: {d}%"
        },
        "legend": {
            "orient": "vertical",
            "left": "left",
            "textStyle": {"fontSize": 12}
        },
        "series": [
            {
                "name": "维度贡献",
                "type": "pie",
                "radius": ["40%", "65%"],
                "center": ["50%", "50%"],
                "data": dimension_data,
                "label": {
                    "show": True,
                    "formatter": "{b}\n{d}%",
                    "fontSize": 11
                },
                "emphasis": {
                    "scale": True,
                    "scaleSize": 10
                },
                "itemStyle": {
                    "borderRadius": 8,
                    "borderColor": "#fff",
                    "borderWidth": 2
                }
            }
        ]
    }
    return option


# =======================
# 5️⃣ 环形图数据
# =======================
pie_data = []
for i, v in counts.items():
    pie_data.append({"name": cluster_profiles[i]["name"], "value": int(v)})

pie_option = {
    "tooltip": {"trigger": "item", "formatter": "{b}<br/>人数：{c} ({d}%)"},
    "color": [cluster_profiles[i]["color"] for i in sorted(cluster_profiles.keys())],
    "series": [{
        "name": "学生群体", "type": "pie", "radius": ["40%", "70%"],
        "itemStyle": {"borderRadius": 8, "borderColor": "#fff", "borderWidth": 2},
        "emphasis": {"scale": True, "scaleSize": 15},
        "label": {"show": True, "formatter": "{b}\n{d}%"},
        "data": pie_data
    }]
}

# =======================
# 6️⃣ 主布局：标签页
# =======================
tab1, tab2, tab3, tab4 = st.tabs(["📊 学生群体画像", "🔍 全局特征贡献排名", "📄 个人诊断报告", "🔬 模型科学性验证"])

# ========== 标签页1：学生群体画像（默认展开） ==========
with tab1:
    col1, col2 = st.columns([3, 2])

    with col1:
        st_echarts(pie_option, height="550px")

    with col2:
        st.markdown("### 📌 群体深度画像")

        for cluster_id, profile in cluster_profiles.items():
            count = counts.get(cluster_id, 0)
            percentage = (count / total_students) * 100 if total_students > 0 else 0

            # 修改这里：expanded=True 让所有卡片默认展开
            with st.expander(f"{profile['icon']} {profile['name']} - {count}人 ({percentage:.1f}%)", expanded=True):
                st.caption(f"*{profile['en_name']}*")
                st.info(profile['summary'])

                st.markdown("**📊 群体特征图谱（Z-Score）**")
                st.caption("💡 正值（绿色）表示高于全校平均水平，负值（红色）表示低于平均水平")

                bar_option = create_bar_chart(cluster_id)
                if bar_option:
                    st_echarts(bar_option, height="350px")
                else:
                    st.warning("暂无数据")

                if cluster_id in zscore_df.index:
                    cluster_z = zscore_df.loc[cluster_id]
                    positive = cluster_z[cluster_z > 0.3].sort_values(ascending=False).head(3)
                    negative = cluster_z[cluster_z < -0.3].sort_values().head(3)

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("**✅ 显著优势**")
                        if len(positive) > 0:
                            for field, value in positive.items():
                                short_name = dimension_names.get(field, field[:12])
                                st.markdown(f"- {short_name}: `+{value:.2f}σ`")
                        else:
                            st.markdown("*无明显优势维度*")

                    with col_b:
                        st.markdown("**⚠️ 明显短板**")
                        if len(negative) > 0:
                            for field, value in negative.items():
                                short_name = dimension_names.get(field, field[:12])
                                st.markdown(f"- {short_name}: `{value:.2f}σ`")
                        else:
                            st.markdown("*无明显短板维度*")

                st.markdown("---")
                st.markdown("**📝 详细分析**")
                st.write(profile['desc'])

        st.markdown("---")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("总样本量", f"{total_students} 人")
        with col_b:
            st.metric("群体类型", f"{len(cluster_profiles)} 类")
        with col_c:
            st.metric("行为维度", f"{len(available_dims)} 个")

# ========== 标签页2：全局特征贡献排名 ==========
with tab2:
    st.markdown("## 🎯 特征重要性分析")
    st.markdown("> 哪些行为维度最能区分不同的学生群体？以下展示了模型计算出的全局特征贡献排名。")

    # 三个维度的说明卡片
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea20, #764ba220); padding: 12px; border-radius: 12px; border-top: 4px solid #667eea;">
            <div style="font-size: 1.2rem; font-weight: bold;">📖 学术硬核维度</div>
            <div style="font-size: 0.75rem; color: #666;">学生画像的“定海神针”</div>
            <div style="font-size: 0.7rem; margin-top: 6px;">综合测评 · 奖学金 · 学科竞赛 · 学业成绩</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4CAF5020, #2196F320); padding: 12px; border-radius: 12px; border-top: 4px solid #4CAF50;">
            <div style="font-size: 1.2rem; font-weight: bold;">🏃 生活自律维度</div>
            <div style="font-size: 0.75rem; color: #666;">画像的“底座”</div>
            <div style="font-size: 0.7rem; margin-top: 6px;">考勤 · 门禁 · 跑步打卡</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #FF980020, #F4433620); padding: 12px; border-radius: 12px; border-top: 4px solid #FF9800;">
            <div style="font-size: 1.2rem; font-weight: bold;">🎯 志趣导向维度</div>
            <div style="font-size: 0.75rem; color: #666;">“心思在哪”</div>
            <div style="font-size: 0.7rem; margin-top: 6px;">毕业去向 · 上网行为 · 社团活动</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 两栏布局：左侧特征重要性条形图，右侧维度贡献饼图
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### 📊 特征重要性排名（Top 10）")
        st.caption("💡 综合得分 = AI注意力权重 × 数据区分度，数值越高表示该特征对区分学生群体的贡献越大")

        # 用 st.bar_chart 作为备选方案
        if len(importance_df) > 0:
            # 准备数据
            chart_data = importance_df.head(10).copy()
            chart_data['feature_cn'] = chart_data['feature_cn']
            chart_data = chart_data.set_index('feature_cn')[['Final_Score']]

            # 使用原生 st.bar_chart 确保能显示
            st.bar_chart(chart_data, height=400, width='stretch')

        else:
            st.warning("暂无特征重要性数据")

    with col_right:
        st.markdown("### 🥧 维度贡献占比")
        st.caption("💡 各维度在整体分类中的相对重要性")

        # 用 st.altair_chart 或简单表格作为备选
        if len(dimension_importance) > 0:
            # 显示维度数据表格
            dim_df = pd.DataFrame({
                '维度': dimension_importance.index,
                '综合贡献': dimension_importance.values
            })
            st.dataframe(dim_df, width='stretch', hide_index=True)

        else:
            st.warning("暂无维度数据")

    # 完整特征排名表格
    st.markdown("---")
    st.markdown("### 📋 完整特征排名")

    # 显示表格
    display_df = importance_df[['feature_cn', 'AI_Attention_Weight', 'Data_Discrimination', 'Final_Score']].copy()
    display_df.columns = ['特征名称', 'AI注意力权重', '数据区分度', '综合得分']
    display_df['综合得分'] = display_df['综合得分'].round(4)
    display_df['AI注意力权重'] = display_df['AI注意力权重'].round(4)
    display_df['数据区分度'] = display_df['数据区分度'].round(4)

    st.dataframe(display_df, width='stretch', height=400)

    # 特征详细解读
    st.markdown("---")
    st.markdown("### 📖 特征详细解读")

    # 按维度分组展示
    for dimension in ["学术硬核维度", "生活自律维度", "志趣导向维度", "其他维度"]:
        dim_features = importance_df[importance_df['dimension'] == dimension]
        if len(dim_features) > 0:
            with st.expander(f"📌 {dimension}（共{len(dim_features)}个特征）", expanded=(dimension != "其他维度")):
                for _, row in dim_features.iterrows():
                    col_a, col_b = st.columns([1, 3])
                    with col_a:
                        st.markdown(f"**{row['feature_cn']}**")
                        st.caption(f"综合得分：{row['Final_Score']:.4f}")
                    with col_b:
                        # 根据特征名称显示解释
                        if any(k in row['feature_cn'] for k in ["综合测评", "Evaluation"]):
                            st.markdown("评价学生学业质量的「终极坐标」，不仅看成绩，还包含德育和能力加分。")
                        elif any(k in row['feature_cn'] for k in ["奖学金", "Scholarship"]):
                            st.markdown("学术卓越的「荣誉勋章」，是区分「领航型」与「活跃型」的关键。")
                        elif any(k in row['feature_cn'] for k in ["竞赛"]):
                            st.markdown("学术竞争力的「试金石」，顶尖学生的标志性特征。")
                        elif any(k in row['feature_cn'] for k in ["线上学习"]):
                            st.markdown("自主学习能力的「风向标」，反映数字化学习素养。")
                        elif any(k in row['feature_cn'] for k in ["成绩"]):
                            st.markdown("最基础的学业表现指标，直接反映学习效果。")
                        elif any(k in row['feature_cn'] for k in ["考勤", "atten"]):
                            st.markdown("学生校园生活的「生命线」，稳定的出勤对应稳定的成绩。")
                        elif any(k in row['feature_cn'] for k in ["门禁", "dorm"]):
                            st.markdown("反映学生的「生活时钟」，规律作息对应自律型，混乱对应游离型。")
                        elif any(k in row['feature_cn'] for k in ["跑步", "run"]):
                            st.markdown("自律意识的「意志力指标」，能坚持跑步的学生学业韧性更强。")
                        elif any(k in row['feature_cn'] for k in ["毕业", "Graduate"]):
                            st.markdown("「目标导向」指标，解释学生当前努力的驱动力来源。")
                        elif any(k in row['feature_cn'] for k in ["上网", "Tags"]):
                            st.markdown("学生注意力的「晴雨表」，过度沉溺预示学术风险。")
                        elif any(k in row['feature_cn'] for k in ["社团"]):
                            st.markdown("社交活跃度的体现，反映学生的精力分配倾向。")
                        else:
                            st.markdown("该特征在群体区分中具有重要作用。")
                    st.divider()

# ========== 标签页3：个人诊断报告 + 风险监控 ==========
with tab3:
    # 统一归因数据列名
    attribution_df = attribution_df.rename(columns={
        '学号': 'student_id',
        '风险分': 'risk_probability',
        '行为偏移量': 'distance_drift',
        '画像分类': 'current_profile',
        '归因结论': 'attribution_conclusion'
    })

    # 删除 cluster 列（如果存在）
    if 'cluster' in radar_df.columns:
        radar_df = radar_df.drop('cluster', axis=1)

    # 设置 attribution_df 索引
    attribution_df = attribution_df.set_index('student_id')

    # 合并风险数据（用于左侧列表）
    risk_merge_df = risk_df.copy()
    if 'current_profile' in attribution_df.columns:
        risk_merge_df['current_profile'] = risk_merge_df['student_id'].map(attribution_df['current_profile'])
    else:
        risk_merge_df['current_profile'] = '未知'


    # 定义风险等级颜色
    def get_risk_level(prob):
        if prob >= 0.8:
            return "🔴 高危"
        elif prob >= 0.5:
            return "🟡 关注"
        else:
            return "🟢 安全"


    def get_risk_color(prob):
        if prob >= 0.8:
            return "#F44336"
        elif prob >= 0.5:
            return "#FF9800"
        else:
            return "#4CAF50"


    # 雷达图可用指标（你数据中实际有的列）
    radar_available = [
        "Evaluation_vector.csv", "Graduate_vector.csv", "Portrait_vector.csv",
        "Scholarship_vector.csv", "Tags_vector.csv", "atten_vector_data.xlsx",
        "dorm_vector_data.xlsx", "run_vector_data.xlsx", "student_behavior_vector.xlsx",
        "体测数据_特征轨迹_vector.xlsx", "体育课_特征向量_vector.xlsx", "体能_vector_variable.csv",
        "作业提交_vector_composite.csv", "四六级_特征向量_vector.xlsx", "图书馆_特征轨迹_vector.xlsx",
        "基本信息_vector_variable.csv", "学生签到_特征向量.csv", "学科竞赛_vector_composite.csv",
        "学籍异动_vector_composite.csv", "成绩_vector_variable.csv", "日常锻炼_vector.xlsx",
        "社团活动_特征向量_vector.xlsx", "线上学习_vector_composite.csv", "讨论记录_特征向量_vector.xlsx",
        "选课_vector_variable.csv"
    ]

    # 只保留实际存在的列
    radar_available = [col for col in radar_available if col in radar_df.columns]

    # 指标中文名映射
    radar_names = {
        "Evaluation_vector.csv": "综合测评",
        "Graduate_vector.csv": "毕业去向",
        "Portrait_vector.csv": "学生画像",
        "Scholarship_vector.csv": "奖学金",
        "Tags_vector.csv": "上网标签",
        "atten_vector_data.xlsx": "考勤",
        "dorm_vector_data.xlsx": "门禁",
        "run_vector_data.xlsx": "跑步",
        "student_behavior_vector.xlsx": "学生行为",
        "体测数据_特征轨迹_vector.xlsx": "体测",
        "体育课_特征向量_vector.xlsx": "体育课",
        "体能_vector_variable.csv": "体能",
        "作业提交_vector_composite.csv": "作业提交",
        "四六级_特征向量_vector.xlsx": "四六级",
        "图书馆_特征轨迹_vector.xlsx": "图书馆",
        "基本信息_vector_variable.csv": "基本信息",
        "学生签到_特征向量.csv": "签到",
        "学科竞赛_vector_composite.csv": "学科竞赛",
        "学籍异动_vector_composite.csv": "学籍异动",
        "成绩_vector_variable.csv": "学业成绩",
        "日常锻炼_vector.xlsx": "日常锻炼",
        "社团活动_特征向量_vector.xlsx": "社团活动",
        "线上学习_vector_composite.csv": "线上学习",
        "讨论记录_特征向量_vector.xlsx": "讨论记录",
        "选课_vector_variable.csv": "选课"
    }

    radar_available_names = [radar_names.get(col, col) for col in radar_available]


    def create_radar_chart(student_id):
        """创建学生个人雷达图"""
        if student_id not in radar_df.index:
            return None

        values = []
        for dim in radar_available:
            val = radar_df.loc[student_id, dim]
            values.append(round(float(val), 2))

        option = {
            "title": {"show": False},
            "tooltip": {"trigger": "item"},
            "radar": {
                "indicator": [{"name": name, "max": 3, "min": -3} for name in radar_available_names],
                "shape": "circle",
                "center": ["50%", "50%"],
                "radius": "60%",
                "name": {"textStyle": {"fontSize": 9}},
                "splitArea": {"areaStyle": {"color": ["rgba(200,200,200,0.2)"]}}
            },
            "series": [{
                "type": "radar",
                "data": [{"value": values, "name": student_id}],
                "areaStyle": {"color": "rgba(102,126,234,0.3)"},
                "lineStyle": {"color": "#667eea", "width": 2},
                "itemStyle": {"color": "#667eea"}
            }]
        }
        return option


    # ========== 布局：左右两栏 ==========
    col_left, col_right = st.columns([2, 3])

    # ========== 左侧：风险红黑榜 ==========
    with col_left:
        st.markdown("### 🚨 全校风险红黑榜")
        st.caption("💡 点击学生卡片查看详情 | ↑ 表示行为正在异常偏移")

        # 搜索框和筛选器放在同一行
        col_search, col_filter = st.columns([2, 1])
        with col_search:
            search_term = st.text_input("🔍 搜索学号", placeholder="输入学号筛选", key="risk_search")
        with col_filter:
            risk_filter = st.selectbox("风险等级", ["全部", "🔴 高危", "🟡 关注", "🟢 安全"])

        # 过滤数据
        filtered_df = risk_merge_df.copy()

        # 学号搜索过滤
        if search_term:
            filtered_df = filtered_df[filtered_df['student_id'].astype(str).str.contains(search_term, na=False)]

        # 风险等级过滤
        if risk_filter == "🔴 高危":
            filtered_df = filtered_df[filtered_df['risk_probability'] >= 0.8]
        elif risk_filter == "🟡 关注":
            filtered_df = filtered_df[
                (filtered_df['risk_probability'] >= 0.5) & (filtered_df['risk_probability'] < 0.8)]
        elif risk_filter == "🟢 安全":
            filtered_df = filtered_df[filtered_df['risk_probability'] < 0.5]

        filtered_df = filtered_df.sort_values('risk_probability', ascending=False)

        # 显示统计
        st.markdown(f"**共 {len(filtered_df)} 名学生**")

        # 存储选中的学号
        if 'selected_student' not in st.session_state:
            st.session_state.selected_student = filtered_df.iloc[0]['student_id'] if len(filtered_df) > 0 else None

        # 可滚动的学生列表
        with st.container(height=450):
            for _, row in filtered_df.iterrows():
                student_id = row['student_id']
                risk_prob = row['risk_probability']
                risk_level = get_risk_level(risk_prob)
                risk_color = get_risk_color(risk_prob)
                profile = row.get('current_profile', '未知')
                drift = row.get('distance_drift', 0)

                # 偏移预警标识
                if drift > 0.05:
                    drift_icon = " ↑"
                elif drift < -0.05:
                    drift_icon = " ↓"
                else:
                    drift_icon = ""

                # 卡片样式（选中的高亮）
                is_selected = (st.session_state.selected_student == student_id)

                if is_selected:
                    button_label = f"✅ {student_id}{drift_icon}  |  {profile}  |  {risk_level} ({risk_prob:.1%})"
                else:
                    button_label = f"{student_id}{drift_icon}  |  {profile}  |  {risk_level} ({risk_prob:.1%})"

                if st.button(button_label, key=f"risk_{student_id}", use_container_width=True):
                    st.session_state.selected_student = student_id
                    st.rerun()
    # ========== 右侧：学生详情面板 ==========
    with col_right:
        if st.session_state.selected_student:
            student_id = st.session_state.selected_student

            # 获取该生的风险信息
            student_risk = risk_df[risk_df['student_id'] == student_id]
            if len(student_risk) > 0:
                risk_prob = student_risk['risk_probability'].values[0]
                drift = student_risk['distance_drift'].values[0]
            else:
                risk_prob = None
                drift = 0
            risk_color = get_risk_color(risk_prob) if risk_prob else "#666"

            # 偏移预警标识
            if drift > 0.05:
                drift_icon = "↑ 风险加剧"
                drift_color = "#F44336"
            elif drift < -0.05:
                drift_icon = "↓ 风险缓解"
                drift_color = "#4CAF50"
            else:
                drift_icon = "→ 状态稳定"
                drift_color = "#888"

            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {risk_color}10, #ffffff);
                padding: 16px;
                border-radius: 12px;
                border-left: 6px solid {risk_color};
                margin-bottom: 16px;
            ">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 1.3rem; font-weight: bold;">📋 学号：{student_id}</span>
                        <span style="margin-left: 12px; font-size: 0.9rem; color: {drift_color};">{drift_icon}</span>
                    </div>
                    <div style="background: {risk_color}; padding: 4px 16px; border-radius: 20px; color: white; font-weight: bold;">
                        {get_risk_level(risk_prob)} ({risk_prob:.1%})
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 雷达图
            st.markdown("### 📊 多维行为雷达图")
            st.caption("💡 正值表示高于全校平均，负值表示低于全校平均")
            radar_option = create_radar_chart(student_id)
            if radar_option:
                st_echarts(radar_option, height="450px")
            else:
                st.warning("暂无雷达图数据")
            # 轨迹偏移散点图
            st.markdown("---")
            st.markdown("### 🎯 轨迹偏移散点图")
            st.caption("💡 横轴：风险分 | 纵轴：行为偏移量 | 红色点表示当前选中学生")

            # 准备散点图数据
            scatter_data = []
            for _, row in risk_merge_df.iterrows():
                scatter_data.append({
                    "name": row['student_id'],
                    "value": [row['risk_probability'], row.get('distance_drift', 0)]
                })

            # 当前选中学生的位置
            current_risk = risk_prob if risk_prob else 0
            current_drift = drift if 'drift' in dir() else 0

            scatter_option = {
                "title": {"show": False},
                "tooltip": {
                    "trigger": "axis",
                    "axisPointer": {"type": "shadow"},
                    "formatter": "{b}<br/>风险分: {c[0]:.2f}<br/>偏移量: {c[1]:.4f}"
                },
                "xAxis": {
                    "name": "风险分",
                    "nameLocation": "middle",
                    "nameGap": 35,
                    "min": 0,
                    "max": 1,
                    "axisLabel": {"formatter": "{value}"}
                },
                "yAxis": {
                    "name": "行为偏移量",
                    "nameLocation": "middle",
                    "nameGap": 40
                },
                "series": [
                    {
                        "name": "全体学生",
                        "type": "scatter",
                        "data": [{"name": d["name"], "value": d["value"]} for d in scatter_data],
                        "symbolSize": 8,
                        "itemStyle": {"color": "#888"},
                        "label": {"show": False}
                    },
                    {
                        "name": "当前学生",
                        "type": "scatter",
                        "data": [{"name": student_id, "value": [current_risk, current_drift]}],
                        "symbolSize": 16,
                        "itemStyle": {"color": "#F44336", "borderColor": "#fff", "borderWidth": 2},
                        "label": {"show": True, "formatter": student_id, "position": "right", "fontSize": 10}
                    }
                ],
                "grid": {
                    "left": "10%",
                    "right": "15%",
                    "top": "10%",
                    "bottom": "10%",
                    "containLabel": True
                }
            }

            st_echarts(scatter_option, height="300px")
            # 归因分析
            st.markdown("---")
            st.markdown("### 📝 归因分析与干预建议")

            if student_id in attribution_df.index:
                attribution_row = attribution_df.loc[student_id]

                col_a, col_b = st.columns(2)
                with col_a:
                    profile = attribution_row.get('current_profile', '未知')
                    st.markdown(f"**当前画像：** {profile}")
                with col_b:
                    drift_val = attribution_row.get('distance_drift', 0)
                    st.markdown(f"**行为偏移量：** {drift_val:.4f}")

                # 归因结论
                conclusion = attribution_row.get('attribution_conclusion', '')
                if conclusion:
                    st.info(f"**归因结论：** {conclusion}")

                # 干预建议
                st.markdown("#### 🎯 干预建议")
                if '综合偏差' in str(conclusion):
                    st.write("建议重点关注该生的综合学业表现，排查是否存在学习方法、时间管理或心理适应问题。")
                elif '行为轨迹发生显著漂移' in str(conclusion):
                    st.write("该生行为模式正在发生异常变化，建议辅导员/班主任主动约谈，了解近期情况。")
                else:
                    st.write("建议结合雷达图中负向偏离较大的维度进行针对性辅导和关注。")
            else:
                st.warning("暂无归因数据")

            # 同时显示个人诊断报告
            if student_id in individual_df['学号'].astype(str).values:
                st.markdown("---")
                st.markdown("### 📄 学业诊断建议书")
                report_row = individual_df[individual_df['学号'].astype(str) == student_id]
                if len(report_row) > 0:
                    report_text = report_row.iloc[0]['individual_report']
                    st.markdown(report_text)
        else:
            # 未选中学生时的占位符
            st.info("👈 请从左侧列表点击学号查看详情")
            st.markdown("""
            <div style="
                background: #f8f9fa;
                padding: 60px 20px;
                border-radius: 12px;
                text-align: center;
                color: #888;
            ">
                <div style="font-size: 3rem;">📊</div>
                <div>点击左侧学生学号</div>
                <div style="font-size: 0.8rem;">查看雷达图、归因分析和干预建议</div>
            </div>
            """, unsafe_allow_html=True)
# ========== 标签页4：模型科学性验证 ==========
with tab4:
    st.markdown("## 🔬 模型科学性验证")
    st.markdown("> ROC 曲线效能分析图 - 验证风险预测模型的准确性")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 读取并显示 ROC 曲线图
        from PIL import Image
        import os
        
        # 尝试读取图片
        roc_path = find_file("A_Distance_ROC.png")
        if os.path.exists(roc_path):
            image = Image.open(roc_path)
            st.image(image, caption="ROC 曲线 - 风险预测模型效能评估", use_container_width=True)
        else:
            st.warning("未找到 A_Distance_ROC.png 文件")
            st.info("""
            **ROC 曲线说明：**
            - 横轴：假正率 (False Positive Rate)
            - 纵轴：真正率 (True Positive Rate)
            - AUC 值越接近 1，模型预测效果越好
            - AUC > 0.9 表示模型具有极高的区分能力
            """)
    
    with col2:
        st.markdown("### 📊 模型评估指标")
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea20, #764ba220);
            padding: 20px;
            border-radius: 12px;
            margin-top: 20px;
        ">
            <div style="text-align: center;">
                <div style="font-size: 2.5rem; font-weight: bold; color: #667eea;">AUC</div>
                <div style="font-size: 1.2rem;">Area Under Curve</div>
                <div style="font-size: 0.8rem; color: #666; margin-top: 10px;">曲线下面积</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("**✅ 模型验证结论：**")
        st.markdown("""
        - ROC 曲线远离对角线，表明模型具有良好的区分能力
        - 高风险学生能被有效识别
        - 模型可用于实际的学业风险预警
        """)
        
        st.markdown("---")
        st.markdown("**📌 对应任务：**")
        st.markdown("- 开发完成度硬性指标")
        st.markdown("- 风险预测模型效能验证")
