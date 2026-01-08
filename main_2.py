import argparse
import os
import sys
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

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run LLM4AD Experiment")
    parser.add_argument('--algorithm', type=str, required=True, choices=algorithm_map.keys())
    parser.add_argument('--problem', type=str, required=True, choices=task_map.keys())
    
    # Api Key String (Optional - dùng để test tay)
    parser.add_argument('--api_key', type=str, default=None, required=False, 
                        help='Mistral API Keys separated by comma')
    
    # Chọn index (0, 1, 2) tương ứng với (API_KEY1, API_KEY3, API_KEY4)
    parser.add_argument('--key_index', type=int, default=-1, required=False,
                        help='Index of the API key to use. 0=API_KEY1, 1=API_KEY3, 2=API_KEY4. Default -1 (use all).')
    
    parser.add_argument('--version', type=str, required=True, choices=['v1', 'v2', 'v3'])
    return parser.parse_args()

def get_kaggle_keys_list():
    """
    Hàm này tự động đi gom các key API_KEY1, API_KEY3, API_KEY4
    từ Kaggle Secrets của bạn.
    """
    found_keys = []
    # Danh sách tên các secret bạn đang có
    target_secret_names = ["API_KEY1", "API_KEY3", "API_KEY4"]
    
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        
        for name in target_secret_names:
            try:
                val = user_secrets.get_secret(name)
                if val:
                    found_keys.append(val)
                    print(f"Loaded {name} from Secrets.")
            except:
                print(f"Secret '{name}' not found or empty.")
                pass
    except ImportError:
        print("Not running in Kaggle environment (kaggle_secrets not found).")
    
    return found_keys

if __name__ == '__main__':
    args = parse_arguments()
    
    ALGORITHM_NAME = args.algorithm
    PROBLEM_NAME = args.problem
    VERSION = args.version
    KEY_INDEX = args.key_index
    
    # --- 1. LẤY DANH SÁCH KEY ---
    full_key_list = []
    
    if args.api_key:
        # Nếu nhập tay qua dòng lệnh
        full_key_list = [k.strip() for k in args.api_key.split(',') if k.strip()]
        print("Using API Keys from command line argument.")
    else:
        # Tự động lấy từ 3 secret của bạn
        print("Retrieving [API_KEY1, API_KEY3, API_KEY4] from Kaggle Secrets...")
        full_key_list = get_kaggle_keys_list()
    
    if not full_key_list:
        print("Error: No API Keys found! Please check your Kaggle Secrets.")
        sys.exit(1)

    # --- 2. XỬ LÝ CHỌN KEY THEO INDEX ---
    final_keys_to_use = []
    
    if KEY_INDEX >= 0:
        if KEY_INDEX < len(full_key_list):
            selected_key = full_key_list[KEY_INDEX]
            final_keys_to_use = [selected_key]
            # In ra để debug (chỉ in 5 ký tự đầu để bảo mật)
            print(f"-> Selected Key Index {KEY_INDEX}: {selected_key[:5]}...***")
        else:
            print(f"Error: --key_index {KEY_INDEX} is out of range! Found only {len(full_key_list)} keys.")
            sys.exit(1)
    else:
        final_keys_to_use = full_key_list
        print(f"-> Using all {len(final_keys_to_use)} keys in rotation.")

    exact_log_dir_name = f"nhv_runtime_reflection/{VERSION}"
    
    print(f"--- Starting Experiment ---")
    print(f"Algorithm: {ALGORITHM_NAME}")
    print(f"Problem:   {PROBLEM_NAME}")
    print(f"Version:   {VERSION}")
    print(f"Log Dir:   {exact_log_dir_name}")
    print(f"---------------------------")

    MethodClass, ProfilerClass = algorithm_map[ALGORITHM_NAME]
    TaskClass = task_map[PROBLEM_NAME]
    
    llm = MistralApi(
        keys=final_keys_to_use,
        model='codestral-latest',
        timeout=60
    )
    
    log_dir = f'logs/{ALGORITHM_NAME}/{PROBLEM_NAME}'
    task = TaskClass 
    
    method = MethodClass(
        llm=llm,
        llm_cluster=llm,
        profiler=ProfilerClass(
            log_dir=log_dir, 
            log_style='complex', 
            result_folder=exact_log_dir_name
        ),
        evaluation=task,
        max_sample_nums=305, 
        max_generations=31,
        pop_size=10, 
        num_samplers=4,
        num_evaluators=4,
        selection_num=2,
        review=True     
    )
    
    method.run()