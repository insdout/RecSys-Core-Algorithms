import tqdm
import pandas as pd
import numpy as np
import copy
import random
from collections import OrderedDict, Counter
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, List, Union, Tuple, Optional


def gen_model_input(df: pd.DataFrame, user_profile: pd.DataFrame, user_col: str,
                    item_profile: pd.DataFrame, item_col: str, seq_max_len: int,
                    padding: str = 'pre', truncating: str = 'pre') -> Dict[str, List[Any]]:
    '''
    Merge user_profile and item_profile, pad history sequence feature

    Args:
        df (pd.DataFrame): The main DataFrame.
        user_profile (pd.DataFrame): User profile DataFrame.
        user_col (str): User column name.
        item_profile (pd.DataFrame): Item profile DataFrame.
        item_col (str): Item column name.
        seq_max_len (int): Maximum length of the history sequence.
        padding (str, optional): Padding method, either 'pre' or 'post'. Defaults to 'pre'.
        truncating (str, optional): Truncating method, either 'pre' or 'post'. Defaults to 'pre'.

    Returns:
        Dict[str, List[Any]]: A dictionary containing input features for the model.
    '''
    df = pd.merge(df, user_profile, on=user_col, how='left')
    df = pd.merge(df, item_profile, on=item_col, how='left')
    for col in df.columns.to_list():
        if col.startswith("hist_"):
            df[col] = pad_sequences(df[col], maxlen=seq_max_len, value=0, padding=padding, truncating=truncating).tolist()
    input_dict = df_to_dict(df)
    return input_dict


def pad_sequences(sequences: List[List[Any]],
                  maxlen: Optional[int] = None,
                  dtype: str = 'int32',
                  padding: str = 'pre',
                  truncating: str = 'pre',
                  value: Any = 0.) -> np.ndarray:
    """
    Pads sequences (list of list) to the ndarray of the same length.

    Args:
        sequences (List[List[Any]]): List of sequences to be padded.
        maxlen (Optional[int]): Maximum length of sequences. If not provided, it will be set to the maximum length of sequences.
        dtype (str, optional): Data type of the resulting array. Defaults to 'int32'.
        padding (str, optional): Padding position, either 'pre' or 'post'. Defaults to 'pre'.
        truncating (str, optional): Truncating position, either 'pre' or 'post'. Defaults to 'pre'.
        value (Any, optional): The value to fill the padding. Defaults to 0.

    Returns:
        np.ndarray: Padded array.
    """
    assert padding in ["pre", "post"], f"Invalid padding={padding}."
    assert truncating in ["pre", "post"], f"Invalid truncating={truncating}."

    if maxlen is None:
        maxlen = max(len(x) for x in sequences)
    arr = np.full((len(sequences), maxlen), value, dtype=dtype)
    for idx, x in enumerate(sequences):
        if len(x) == 0:
            continue
        if truncating == 'pre':
            trunc = x[-maxlen:]
        else:
            trunc = x[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)

        if padding == 'pre':
            arr[idx, -len(trunc):] = trunc
        else:
            arr[idx, :len(trunc)] = trunc
    return arr


def df_to_dict(data: pd.DataFrame) -> Dict[str, Union[np.ndarray, List[Any]]]:
    """
    Convert the DataFrame to a dict type input that the network can accept.

    Args:
        data (pd.DataFrame): Datasets of type DataFrame.

    Returns:
        Dict[str, Union[np.ndarray, List[Any]]]: The converted dict, which can be used directly into the input network.
    """
    data_dict = data.to_dict('list')
    for key in data.keys():
        data_dict[key] = np.array(data_dict[key])
    return data_dict


def negative_sample(items_cnt_order: Dict[Union[str, int], int], ratio: int, method_id: int = 0) -> List[Union[str, int]]:
    """
    Negative Sample method for matching model
    reference: https://github.com/wangzhegeek/DSSM-Lookalike/blob/master/utils.py

    Args:
        items_cnt_order (Dict[Union[str, int], int]): The item count dict, where keys (item) are sorted by value (count) in reverse order.
        ratio (int): Negative sample ratio, should be >= 1.
        method_id (int, optional):
            {
                0: "random sampling",
                1: "popularity sampling method used in word2vec",
                2: "popularity sampling method by `log(count+1)+1e-6`",
                3: "tencent RALM sampling"
            }. Defaults to 0.

    Returns:
        List[Union[str, int]]: Sampled negative item list
    """
    items_set = [item for item, count in items_cnt_order.items()]
    if method_id == 0:
        neg_items = np.random.choice(items_set, size=ratio, replace=True)
    elif method_id == 1:
        p_sel = {item: count**0.75 for item, count in items_cnt_order.items()}
        p_value = np.array(list(p_sel.values())) / sum(p_sel.values())
        neg_items = np.random.choice(items_set, size=ratio, replace=True, p=p_value)
    elif method_id == 2:
        p_sel = {item: np.log(count + 1) + 1e-6 for item, count in items_cnt_order.items()}
        p_value = np.array(list(p_sel.values())) / sum(p_sel.values())
        neg_items = np.random.choice(items_set, size=ratio, replace=True, p=p_value)
    elif method_id == 3:
        p_sel = {item: (np.log(k + 2) - np.log(k + 1) / np.log(len(items_cnt_order) + 1)) for item, k in items_cnt_order.items()}
        p_value = np.array(list(p_sel.values())) / sum(p_sel.values())
        neg_items = np.random.choice(items_set, size=ratio, replace=False, p=p_value)
    else:
        raise ValueError("method id should be in (0, 1, 2, 3)")
    return neg_items


def generate_seq_feature_match(data: pd.DataFrame,
                               user_col: str,
                               item_col: str,
                               time_col: str,
                               item_attribute_cols: List[str] = None,
                               sample_method: int = 0,
                               mode: int = 0,
                               neg_ratio: int = 0,
                               min_item: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate sequence feature and negative sample for match.

    Args:
        data (pd.DataFrame): The raw data.
        user_col (str): The col name of user_id.
        item_col (str): The col name of item_id.
        time_col (str): The col name of timestamp.
        item_attribute_cols (List[str], optional): The other attribute cols of item which you want to generate sequence feature. Defaults to `[]`.
        sample_method (int, optional): The negative sample method `{
            0: "random sampling",
            1: "popularity sampling method used in word2vec",
            2: "popularity sampling method by `log(count+1)+1e-6`",
            3: "tencent RALM sampling"}`.
            Defaults to 0.
        mode (int, optional): The training mode, `{0:point-wise, 1:pair-wise, 2:list-wise}`. Defaults to 0.
        neg_ratio (int, optional): Negative sample ratio, should be >= 1. Defaults to 0.
        min_item (int, optional): The min item each user must have. Defaults to 0.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Split train and test data with sequence features.
    """
    if item_attribute_cols is None:
        item_attribute_cols = []
    if mode == 2:  # list-wise learning
        assert neg_ratio > 0, 'neg_ratio must be greater than 0 when list-wise learning'
    elif mode == 1:  # pair-wise learning
        neg_ratio = 1
    data.sort_values(time_col, inplace=True)
    train_set, test_set = [], []
    n_cold_user = 0

    items_cnt = Counter(data[item_col].tolist())
    items_cnt_order = OrderedDict(sorted((items_cnt.items()), key=lambda x: x[1], reverse=True))  # item_id:item count
    neg_list = negative_sample(items_cnt_order, ratio=data.shape[0] * neg_ratio, method_id=sample_method)
    neg_idx = 0
    for uid, hist in tqdm.tqdm(data.groupby(user_col), desc='generate sequence features'):
        pos_list = hist[item_col].tolist()
        if len(pos_list) < min_item:
            n_cold_user += 1
            continue

        for i in range(1, len(pos_list)):
            hist_item = pos_list[:i]
            sample = [uid, pos_list[i], hist_item, len(hist_item)]
            if len(item_attribute_cols) > 0:
                for attr_col in item_attribute_cols:
                    sample.append(hist[attr_col].tolist()[:i])
            if i != len(pos_list) - 1:
                if mode == 0:  # point-wise, include label 0 and 1
                    last_col = "label"
                    train_set.append(sample + [1])
                    for _ in range(neg_ratio):
                        sample[1] = neg_list[neg_idx]
                        neg_idx += 1
                        train_set.append(sample + [0])
                elif mode == 1:  # air-wise, include one negative item
                    last_col = "neg_items"
                    for _ in range(neg_ratio):
                        sample_copy = copy.deepcopy(sample)
                        sample_copy.append(neg_list[neg_idx])
                        neg_idx += 1
                        train_set.append(sample_copy)
                elif mode == 2:  # list-wise, include neg_ratio negative items
                    last_col = "neg_items"
                    sample.append(neg_list[neg_idx: neg_idx + neg_ratio])
                    neg_idx += neg_ratio
                    train_set.append(sample)
                else:
                    raise ValueError("mode should in (0,1,2)")
            else:
                test_set.append(sample + [1])

    random.shuffle(train_set)
    random.shuffle(test_set)

    print("n_train: %d, n_test: %d" % (len(train_set), len(test_set)))
    print("%d cold start users dropped " % (n_cold_user))

    attr_hist_col = ["hist_" + col for col in item_attribute_cols]
    df_train = pd.DataFrame(train_set,
                            columns=[user_col, item_col, "hist_" + item_col, "histlen_" + item_col] + attr_hist_col + [last_col])
    df_test = pd.DataFrame(test_set,
                           columns=[user_col, item_col, "hist_" + item_col, "histlen_" + item_col] + attr_hist_col + [last_col])

    return df_train, df_test


class TorchDataset(Dataset):

    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.x.items()}, self.y[index]

    def __len__(self):
        return len(self.y)


class PredictDataset(Dataset):

    def __init__(self, x):
        super().__init__()
        self.x = x

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.x.items()}

    def __len__(self):
        return len(self.x[list(self.x.keys())[0]])


class MatchDataGenerator(object):

    def __init__(self, x, y=[]):
        super().__init__()
        if len(y) != 0:
            self.dataset = TorchDataset(x, y)
        else:  # For pair-wise model, trained without given label
            self.dataset = PredictDataset(x)

    def generate_dataloader(self, x_test_user, x_all_item, batch_size, num_workers=8):
        train_dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = PredictDataset(x_test_user)

        # shuffle = False to keep same order as ground truth
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        item_dataset = PredictDataset(x_all_item)
        item_dataloader = DataLoader(item_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_dataloader, test_dataloader, item_dataloader
