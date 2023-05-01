from pathlib import Path
this_file = Path(__file__).resolve().absolute()
this_directory = this_file.parent
project_directory = this_directory.parent
data_directory = project_directory / 'data'
import sys
if str(project_directory) not in sys.path:
    sys.path.append(str(project_directory))


from datasets.aliexpress import AliExpressDataset
from datasets.mmoe_synthetic import SynDataset
from datasets.abstract_dataset import MultitaskDataset

import joblib
memory = joblib.Memory('./joblib_tmp', verbose=1)

@memory.cache
def get_dataset(type, path)->MultitaskDataset:
    """数据集工厂

    Args:
        type (str): 数据集的类型，用于决定使用什么子类构造数据集对象。比如AliExpress类型需要用AliExpressDataset去构造。
        path (str): 数据集的路径。如果是绝对路径则无问题。如果是相对路径，默认是相对于项目文件夹的路径。如果找不到，则在项目/data文件夹下寻找。

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    path = Path(path)
    if not Path(path).is_absolute():
        rel_path = project_directory / path
        if not rel_path.exists():
            rel_path = data_directory / path
            if not rel_path.exists():
                raise FileNotFoundError('cannot find dataset: ' + str(path))
        path = rel_path.resolve().absolute()
    if 'AliExpressDataset' in type:
        return AliExpressDataset(path)
    elif 'SynDataset' in type:
        return SynDataset(path)
    else:
        raise ValueError('unknown dataset name: ' + type)
    
    # exec(f'res = {type}(path)')
    # return res
    # return type(path)
    
def get_dataset_by_yaml(type, path):
    raise NotImplementedError