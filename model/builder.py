from model.unidm_di import UniDM_DataImputation
from model.unidm_dt import UniDM_DataTransformation
from model.unidm_em import UniDM_EntityResolution


def build_model(args, logger):
    """
    Builds the model 
    """

    if args.task == "data_imputation":
        UniDM = UniDM_DataImputation(args, logger)
    elif args.task == "data_transformation":
        UniDM = UniDM_DataTransformation(args, logger)
    elif args.task == "entity_resolution":
        UniDM = UniDM_EntityResolution(args, logger)
    else:
        raise ValueError('Unrecognized Task:%s.' % args.task)

    return UniDM