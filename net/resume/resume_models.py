import os.path
import shutil


def resume_models(path_best_models: dict,
                  path_resume_models: dict):

    # copy best-model AUFROC [0, 1]
    if os.path.exists(path_resume_models['AUFROC']['[0, 1]']):
        shutil.copy2(src=path_resume_models['AUFROC']['[0, 1]'],
                     dst=path_best_models['AUFROC']['[0, 1]'])

    print("resume models: COMPLETE")
