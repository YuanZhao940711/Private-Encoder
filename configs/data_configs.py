from configs import transforms_config


DATASETS = {
	# 在训练pSp Enoder的任务中，target 和 source image是统一类的
	'ffhq_encode': transforms_config.EncodeTransforms,
	'ffhq_frontalize': transforms_config.FrontalizationTransforms,
	'celeba_encode': transforms_config.EncodeTransforms,
	'celebs_sketch_to_face': transforms_config.SketchToImageTransforms,
	'celebs_seg_to_face': transforms_config.SegToImageTransforms,
	'celebs_super_resolution': transforms_config.SuperResTransforms,
}
