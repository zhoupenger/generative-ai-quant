import os

def get_project_root():
    """返回项目的根目录。"""
    return os.path.dirname(os.getcwd())

def get_env_path(env_name):
    """返回特定的子目录地址。"""
    parent_path = os.path.dirname(os.getcwd())
    env_path = parent_path + '/' + env_name + '.env'
    
    return env_path

def get_prompt_from_file(file_name):
    parent_path = os.path.dirname(os.getcwd())
    prompt_path = parent_path + '/prompts/' + file_name + '.txt'

    with open(prompt_path, 'r') as fp:
        prompt_str = fp.read().strip().replace('\n', '')
    return prompt_str