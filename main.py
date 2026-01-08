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

from llm4ad.method.momcts import MOMCTS_AHD, MOMCTSProfiler
from llm4ad.method.meoh import MEoH, MEoHProfiler
from llm4ad.method.eoh import EoH, EoHProfiler
from llm4ad.method.reevo import ReEvo, ReEvoProfiler
# from llm4ad.method.hsevo import HSEvo, HSEvoProfiler
from llm4ad.method.nsga2 import NSGA2, NSGA2Profiler
from llm4ad.method.mpage import MPaGEProfiler, MPaGE
from llm4ad.method.moead import MOEAD, MOEADProfiler
import os
from dotenv import load_dotenv

load_dotenv()

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

# Change variable here
ALGORITHM_NAME = 'eoh' # Could also be 'MEoH' or 'NSGA2'
PROBLEM_NAME = ["bi_kp_gls", "bi_kp_aco","tri_tsp_gls", "tri_tsp_aco", "bi_tsp_aco","bi_tsp_gls"]   # Could also be "tsp_semo, bi_kp, bi_cvrp"
exact_log_dir_name = "nhv_runtime_reflection/v1" # must be unique here
api_key = os.getenv('API_KEY1') # change APIKEY1, APIKEY2, APIKEY3

if __name__ == '__main__':
    
    # Khởi tạo LLM ngoài vòng lặp để tiết kiệm tài nguyên
    llm = MistralApi(
        keys=api_key,
        model='codestral-latest',
        timeout=60
    )

    # Lặp qua từng bài toán trong danh sách
    for prob_name in PROBLEM_NAME:
        print(f"--- Đang bắt đầu thực hiện bài toán: {prob_name} ---")
        
        # Tạo đường dẫn log riêng cho mỗi bài toán
        log_dir = f'logs/{ALGORITHM_NAME}/{prob_name}'
        
        MethodClass, ProfilerClass = algorithm_map[ALGORITHM_NAME]
        task = task_map[prob_name] # Lấy bài toán cụ thể từ map
        
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
        
        try:
            method.run()
            print(f"--- Hoàn thành bài toán: {prob_name} ---")
        except Exception as e:
            print(f"Lỗi khi chạy bài toán {prob_name}: {e}")
            continue # Tiếp tục bài toán tiếp theo nếu bài hiện tại lỗi