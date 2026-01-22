import argparse
import os
import sys
import gc
from dotenv import load_dotenv

# --- Import các module đánh giá (GIỮ NGUYÊN) ---
from llm4ad.task.optimization.bi_tsp_semo import BITSPEvaluation
from llm4ad.task.optimization.bi_kp import BIKPEvaluation
from llm4ad.task.optimization.bi_kp_gls import BIKPGLSEvaluation
from llm4ad.task.optimization.bi_kp_aco import BIKPACOEvaluation
from llm4ad.task.optimization.bi_cvrp import BICVRPEvaluation
from llm4ad.task.optimization.bi_cvrp_gls import BICVRPGLSEvaluation  
from llm4ad.task.optimization.bi_cvrp_aco import BICVRPACOEvaluation 
from llm4ad.task.optimization.tri_tsp_semo import TRITSPEvaluation
from llm4ad.task.optimization.tri_tsp_gls import TRITSPGLSEvaluation
from llm4ad.task.optimization.tri_tsp_aco import TRITSPACOEvaluation
from llm4ad.task.optimization.bi_tsp_aco import TSPACOEvaluation
from llm4ad.task.optimization.bi_tsp_gls import TSPGLSEvaluation

from llm4ad.tools.llm.llm_api_codestral import MistralApi

# --- Import các phương pháp (GIỮ NGUYÊN) ---
from llm4ad.method.momcts import MOMCTS_AHD, MOMCTSProfiler
from llm4ad.method.meoh import MEoH, MEoHProfiler
from llm4ad.method.eoh import EoH, EoHProfiler
from llm4ad.method.reevo import ReEvo, ReEvoProfiler
from llm4ad.method.nsga2 import NSGA2, NSGA2Profiler
from llm4ad.method.mpage import MPaGEProfiler, MPaGE
from llm4ad.method.moead import MOEAD, MOEADProfiler

load_dotenv()

# --- Cấu hình Maps (GIỮ NGUYÊN) ---
algorithm_map = {
    'momcts': (MOMCTS_AHD, MOMCTSProfiler),
    'meoh': (MEoH, MEoHProfiler),
    'eoh': (EoH, EoHProfiler),
    'reevo': (ReEvo, ReEvoProfiler),
    'nsga2': (NSGA2, NSGA2Profiler),
    'mpage': (MPaGE, MPaGEProfiler),
    'moead': (MOEAD, MOEADProfiler)
}

task_map = {
    "tsp_semo": BITSPEvaluation(),
    "bi_kp": BIKPEvaluation(),
    "bi_cvrp": BICVRPEvaluation(),
    "tri_tsp": TRITSPEvaluation(),
    "bi_tsp_aco": TSPACOEvaluation(),
    "bi_tsp_gls": TSPGLSEvaluation(),
    "tri_tsp_gls": TRITSPGLSEvaluation(),
    "tri_tsp_aco": TRITSPACOEvaluation(),
    "bi_cvrp_gls": BICVRPGLSEvaluation(),
    "bi_cvrp_aco": BICVRPACOEvaluation(),
    "bi_kp_gls": BIKPGLSEvaluation(),
    "bi_kp_aco": BIKPACOEvaluation(), 
}

SEQUENCE_PROBLEMS = [
    "tri_tsp_gls",
    "tri_tsp_aco",
    "bi_tsp_aco",
    "bi_tsp_gls",
    "bi_cvrp_gls",
    "bi_cvrp_aco",
    "bi_kp_gls",
    "bi_kp_aco"
]

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run LLM4AD Experiment")
    parser.add_argument('--algorithm', type=str, required=True, choices=algorithm_map.keys())
    
    valid_problems = list(task_map.keys()) + ['all']
    parser.add_argument('--problem', type=str, required=True, choices=valid_problems)
    
    parser.add_argument('--api_key', type=str, default=None, required=False, 
                        help='Mistral API Keys separated by comma')
    
    parser.add_argument('--key_index', type=int, default=-1, required=False,
                        help='Index of the API key to use. Default -1 (use all).')
    
    parser.add_argument('--version', type=str, required=True, choices=['v1', 'v2', 'v3'])
    return parser.parse_args()

def get_kaggle_keys_list():
    found_keys = []
    target_secret_names = ["API_KEY1", "API_KEY3", "API_KEY4", "API_KEY5", "API_KEY6"]
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        for name in target_secret_names:
            try:
                val = user_secrets.get_secret(name)
                if val: found_keys.append(val)
            except: pass
    except ImportError: pass
    return found_keys

if __name__ == '__main__':
    args = parse_arguments()
    
    # 1. XỬ LÝ API KEY
    full_key_list = []
    if args.api_key:
        full_key_list = [k.strip() for k in args.api_key.split(',') if k.strip()]
        print("Using API Keys from command line.")
    else:
        print("Retrieving [API_KEY1, API_KEY3, API_KEY4] from Kaggle Secrets...")
        full_key_list = get_kaggle_keys_list()
    
    if not full_key_list:
        print("Error: No API Keys found!")
        sys.exit(1)

    # Biến này sẽ chứa 1 chuỗi String duy nhất (key được chọn)
    selected_api_key_string = ""

    if args.key_index >= 0:
        if args.key_index < len(full_key_list):
            selected_api_key_string = full_key_list[args.key_index]
            print(f"-> Selected Key Index {args.key_index}: {selected_api_key_string[:5]}...***")
        else:
            print(f"Error: Key index out of range.")
            sys.exit(1)
    else:
        # Nếu chọn -1 (tất cả), ta lấy cái đầu tiên để tránh lỗi Pydantic
        # Vì library này có vẻ không hỗ trợ list key rotation
        print("-> Using the FIRST key found (List input caused validation error).")
        selected_api_key_string = full_key_list[0]

    # 2. XÁC ĐỊNH BÀI TOÁN
    problems_to_run = []
    if args.problem == 'all':
        problems_to_run = SEQUENCE_PROBLEMS
        print(f"Mode: SEQUENCE -> Will run {len(problems_to_run)} tasks consecutively.")
    else:
        problems_to_run = [args.problem]
        print(f"Mode: SINGLE -> Will run {args.problem} only.")

    ALGORITHM_NAME = args.algorithm
    VERSION = args.version
    MethodClass, ProfilerClass = algorithm_map[ALGORITHM_NAME]

    # 3. VÒNG LẶP CHẠY BÀI TOÁN
    for i, p_name in enumerate(problems_to_run):
        print("\n" + "="*50)
        print(f"STARTING TASK {i+1}/{len(problems_to_run)}: {p_name}")
        print("="*50)
        
        method = None
        llm = None

        log_dir = f'logs/{ALGORITHM_NAME}/{p_name}'
        exact_log_dir_name = f"nhv_runtime_reflection_tsp_aco/s200/{VERSION}"

        try:
            # FIX LỖI: Truyền thẳng String vào api_key, không dùng list
            llm = MistralApi(
                keys=selected_api_key_string, # Truyền String!
                model='codestral-latest',
                timeout=60
            )

            TaskClass = task_map[p_name]
            
            method = MethodClass(
                llm=llm,
                llm_cluster=llm,
                profiler=ProfilerClass(
                    log_dir=log_dir, 
                    log_style='complex', 
                    result_folder=exact_log_dir_name
                ),
                evaluation=TaskClass,
                max_sample_nums=305, 
                max_generations=31,
                pop_size=10, 
                num_samplers=2,
                num_evaluators=2,
                selection_num=2,
                review=True     
            )
            
            method.run()
            print(f">>> FINISHED TASK: {p_name}")

        except Exception as e:
            # In đầy đủ traceback để dễ debug nếu còn lỗi
            import traceback
            traceback.print_exc()
            print(f"!!! CRITICAL ERROR in task {p_name}: {e}")
            # continue 
        
        if method is not None:
            del method
        if llm is not None:
            del llm
            
        gc.collect() 
        print(f">>> Cleaned up memory. Moving to next task...")

    print("\n" + "="*50)
    print("ALL TASKS COMPLETED SUCCESSFULLY!")
    print("="*50)
