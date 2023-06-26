from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

class DetectronModule(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused")
        print(self.metadata)
        self.device = cfg.device
        self.instance_mode = instance_mode
        self.parallel = parallel

        self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image):
        vis_output = None

        predictions = self.predictor(image)

        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)

        if "panoptic_seg" in predictions:
            vis_output = visualizer.draw_sem_seg(predictions["sem_seg"].argmax(dim=0).to("cpu"))
        if "instances" in predictions:
            instances = predictions["instances"]
            vis_output = visualizer.draw_instance_predictions(predictions=instances.to("cpu"))
        return predictions, vis_output