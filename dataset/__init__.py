from dataset.data_target import MetaDataset
from dataset.data_util import SystemDataLoader


def dataset_constructor(args):
    datasource = args.datasource
    if datasource in ["chembl", "bdb"]:
        dataset = MetaDataset
    else:
        raise ValueError(f"model {datasource} is not supported")

    return SystemDataLoader(args, dataset)

