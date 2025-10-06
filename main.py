from llm4ad.task.optimization.bi_tsp_semo import BITSPEvaluation
from llm4ad.task.optimization.bi_kp import BIKPEvaluation
from llm4ad.tools.llm.llm_api_gemini import GeminiApi
from llm4ad.tools.llm.llm_api_codestral import MistralApi
from llm4ad.tools.profiler import ProfilerBase
from llm4ad.method.momcts import MOMCTS_AHD, MOMCTSProfiler
from llm4ad.method.meoh import MEoH, MEoHProfiler
from llm4ad.method.nsga2 import NSGA2, NSGA2Profiler
from llm4ad.method.mpage import EoHProfiler, MPaGE
import os
from dotenv import load_dotenv

load_dotenv()


algorithm_map = {
    'momcts': (MOMCTS_AHD, MOMCTSProfiler),
    'meoh': (MEoH, MEoHProfiler),
    'nsga2': (NSGA2, NSGA2Profiler),
    'mpage': (MPaGE, EoHProfiler)
}

ALGORITHM_NAME = 'momcts'  # Could also be 'MEoH' or 'NSGA2'
PROBLEM_NAME = "tsp_semo" # Could also be "tsp_semo, bi_kp"

MethodClass, ProfilerClass = algorithm_map[ALGORITHM_NAME]
log_dir = f'logs/{ALGORITHM_NAME}/{PROBLEM_NAME}'
exact_log_dir_name = "codestral_300_v6" # must be unique here

if __name__ == '__main__':
    llm = MistralApi(
        keys="M8cikiCWLLo0nEARlNY88jaBauCyaXWB",
        model='codestral-latest',
        timeout=60
    )
    task = BITSPEvaluation() # BITSPEvaluation
    method = MethodClass(
        llm=llm,
        llm_cluster=llm,
        profiler=ProfilerClass(log_dir=log_dir, log_style='complex', result_folder = exact_log_dir_name),
        evaluation=task,
        max_sample_nums=300, # max_sample_nums : terminate after evaluating max_sample_nums functions (no matter the function is valid or not) or reach 'max_generations',
        max_generations=30,
        pop_size=10, # 20
        num_samplers=4,
        num_evaluators=4,
        selection_num=2 # change to 5 for meoh  
        )
    method.run()