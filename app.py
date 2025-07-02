from flask import Flask, render_template, request, redirect
import pandas as pd
import math

app = Flask(__name__)

# Load updated CSV data
uc_bp_cp_df = pd.read_csv("usecase_blueprint_component_full.csv")
component_specs_df = pd.read_csv("components_sizing_v2.csv")

# Utility Functions
def safe_float(val):
    try:
        return float(val)
    except:
        return 0.0

def compute_units(metric, workload_type, user_inputs, base_capacity):
    concurrency = user_inputs["concurrency"]
    tokens = user_inputs["tokens"]
    queries = user_inputs["queries"]
    frames = user_inputs["frames"]

    if base_capacity == 0:
        return 1

    if workload_type == "Token Based":
        if metric == "Tokens/sec":
            return math.ceil((concurrency * tokens) / base_capacity)
        elif metric == "Vectors Encoded/sec":
            return math.ceil(tokens / base_capacity)
        elif metric == "Vector Queries/sec":
            return math.ceil(queries / base_capacity)
        elif metric == "Images/sec":
            return math.ceil(frames / base_capacity)
    elif workload_type == "Non Token Based":
        if metric == "Users Supported":
            return math.ceil(concurrency / base_capacity)
        elif metric == "MB Processed/sec":
            return math.ceil(frames / base_capacity)
        elif metric == "Vector Queries/sec":
            return math.ceil(queries / base_capacity)

    return 1

# Modularized Recommendation Engine
def filter_components_by_usecase(selected_ucs):
    return uc_bp_cp_df[uc_bp_cp_df["Usecase"].isin(selected_ucs)]["Component"].unique()

def get_component_row(comp):
    return component_specs_df[component_specs_df["Component"].str.strip().str.lower() == comp.strip().lower()]

def calculate_gpu_units(row, metric, workload_type, user_inputs):
    return {
        "Gaudi 3": compute_units(metric, workload_type, user_inputs, safe_float(row.get("Per Unit Gaudi 3", 1))),
        "Gaudi 2": compute_units(metric, workload_type, user_inputs, safe_float(row.get("Per Unit Gaudi 2", 1))),
        "A100": compute_units(metric, workload_type, user_inputs, safe_float(row.get("Per Unit GPU (A100)", 1))),
        "H100": compute_units(metric, workload_type, user_inputs, safe_float(row.get("Per Unit GPU (H100)", 1)))
    }

def determine_cpu_base_capacity(row, workload_type):
    if workload_type == "Token Based":
        return safe_float(row.get("Per Unit Gaudi 3", 1))
    scaling_formula = row.get("Scaling Formula", "")
    base_capacity = 75
    if "Users" in scaling_formula:
        try:
            base_capacity = float(scaling_formula.split('=')[1].split('Users')[0].strip())
        except:
            pass
    return base_capacity

def calculate_cpu_memory(row, workload_type, units_cpu):
    if workload_type == "Token Based":
        vcpu = safe_float(row.get("vCPU", 0))
        ram = safe_float(row.get("RAM", 0))
    else:
        base_vcpu = safe_float(row.get("vCPU", 0))
        base_ram = safe_float(row.get("RAM", 0))
        vcpu = int(base_vcpu * units_cpu)
        ram = int(base_ram * units_cpu)
    return int(vcpu), int(ram)

def assemble_recommendation(comp, workload_type, metric, vcpu, ram, gpu_units):
    return {
        "Component": comp,
        "Applicable": "Yes",
        "Workload Type": workload_type,
        "Metric": metric,
        "Standard vCPUs": vcpu,
        "Memory (RAM)": ram,
        "Gaudi 3 Cores": round(gpu_units["Gaudi 3"], 1),
        "Gaudi 2 Cores": round(gpu_units["Gaudi 2"], 1),
        "A100 GPUs": round(gpu_units["A100"], 1),
        "H100 GPUs": round(gpu_units["H100"], 1)
    }

def generate_recommendations(selected_ucs, user_inputs):
    all_components = filter_components_by_usecase(selected_ucs)
    total_vcpus = 0
    total_memory = 0
    total_gpu_units = {
        "Gaudi 2": 0,
        "Gaudi 3": 0,
        "A100": 0,
        "H100": 0
    }
    recommendations = []

    for comp in all_components:
        row_df = get_component_row(comp)
        if not row_df.empty:
            row = row_df.iloc[0]
            workload_type = row.get("Workload Type", "")
            metric = row.get("Metric", "")

            if workload_type == "Token Based":
                gpu_units = calculate_gpu_units(row, metric, workload_type, user_inputs)
            else:
                gpu_units = {k: 0 for k in total_gpu_units}

            base_capacity = determine_cpu_base_capacity(row, workload_type)
            units_cpu = int(compute_units(metric, workload_type, user_inputs, base_capacity))
            vcpu, ram = calculate_cpu_memory(row, workload_type, units_cpu)

            for k in total_gpu_units:
                total_gpu_units[k] += gpu_units[k]

            total_vcpus += vcpu
            total_memory += ram
            recommendation = assemble_recommendation(comp, workload_type, metric, vcpu, ram, gpu_units)
            recommendations.append(recommendation)

    return recommendations, total_vcpus, total_memory, total_gpu_units

# Flask Routes
@app.route("/")
def index():
    usecases = sorted(uc_bp_cp_df['Usecase'].unique())
    blueprints = sorted(uc_bp_cp_df['Blueprint'].unique())
    components = sorted(uc_bp_cp_df['Component'].unique())
    return render_template("index.html", usecases=usecases, all_blueprints=blueprints, all_components=components)

@app.route("/add_usecase", methods=["POST"])
def add_usecase():
    new_uc = request.form.get("usecase_name")
    selected_blueprints = request.form.getlist("blueprints")
    new_rows = []
    for bp in selected_blueprints:
        comps = uc_bp_cp_df[uc_bp_cp_df['Blueprint'] == bp]['Component'].unique()
        for comp in comps:
            new_rows.append({'Usecase': new_uc, 'Blueprint': bp, 'Component': comp})
    pd.DataFrame(new_rows).to_csv("usecase_blueprint_component_full.csv", mode='a', index=False, header=False)
    return redirect("/")

@app.route("/add_blueprint", methods=["POST"])
def add_blueprint():
    new_bp = request.form.get("blueprint_name")
    selected_comps = request.form.getlist("components")
    existing_ucs = uc_bp_cp_df['Usecase'].unique()
    new_rows = []
    for uc in existing_ucs:
        for comp in selected_comps:
            new_rows.append({'Usecase': uc, 'Blueprint': new_bp, 'Component': comp})
    pd.DataFrame(new_rows).to_csv("usecase_blueprint_component_full.csv", mode='a', index=False, header=False)
    return redirect("/")

@app.route("/result", methods=["POST"])
def result():
    selected_ucs = request.form.getlist("usecase")
    llm_size = int(request.form.get("llm_size") or 0)
    concurrency = int(request.form.get("concurrency") or 0)
    tokens = int(request.form.get("tokens_per_user") or 0)
    queries = int(request.form.get("query_rate") or 0)
    frames = int(request.form.get("frames_per_sec") or 0)
    hardware_pref = request.form.get("preferred_hardware")

    user_inputs = {
        "concurrency": concurrency,
        "tokens": tokens,
        "queries": queries,
        "frames": frames
    }

    recommendations, total_vcpus, total_memory, gpu_units = generate_recommendations(selected_ucs, user_inputs)

    processor_descriptions = {
        "Gaudi 2": "Gaudi 2 is ideal for cost-efficient, high-performance inference and RAG workloads.",
        "Gaudi 3": "Gaudi 3 excels in multimodal and LLM tasks with very high performance at medium cost.",
        "A100": "A100 is well-suited for training and inference with balanced performance.",
        "H100": "H100 is recommended for cutting-edge multimodal and agentic AI workloads requiring top-tier performance."
    }

    selected_processor_desc = processor_descriptions.get(hardware_pref, "Selected processor is suitable for general workloads.")

    metrics = [rec["Metric"].lower() for rec in recommendations]
    workload_recommendation = []

    if any("training" in m or "inference" in m for m in metrics) or \
       any("multimodal" in m or "agentic" in m for m in metrics):
        if any("training" in m or "inference" in m for m in metrics):
            workload_recommendation.append("A100")
        if any("multimodal" in m or "agentic" in m for m in metrics):
            workload_recommendation.append("H100")
    else:
        if any("rag" in m or "tokens" in m or "images" in m for m in metrics):
            workload_recommendation.append("Gaudi 2")
        if any("llm" in m for m in metrics):
            workload_recommendation.append("Gaudi 3")

    suggested_line = f"Based on your workload types, we would recommend considering: {', '.join(set(workload_recommendation))}."

    explanation = f"{selected_processor_desc} Based on your inputs (LLM={llm_size}B, concurrency={concurrency}, tokens/sec={tokens}).\n"
    explanation += f"{suggested_line}\n\n"
    explanation += "📌 **Processor Guidance**:\n"
    explanation += "- **Gaudi2**: Low cost, high performance for Inference and RAG workloads. ⭐⭐⭐⭐⭐\n"
    explanation += "- **Gaudi3**: Medium cost, very high performance for Multimodal and LLM workloads. ⭐⭐⭐⭐\n"
    explanation += "- **A100**: High cost, high performance for Training and Inference. ⭐⭐⭐\n"
    explanation += "- **H100**: Very high cost, very high performance for Multimodal and Agentic AI. ⭐⭐\n"

    return render_template("result.html", recommendations=recommendations,
                           total_vcpus=int(total_vcpus),
                           total_memory=int(total_memory),
                           hardware_explanation=explanation,
                           gaudi2_units=gpu_units["Gaudi 2"],
                           gaudi3_units=gpu_units["Gaudi 3"],
                           a100_units=gpu_units["A100"],
                           h100_units=gpu_units["H100"])

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
)
