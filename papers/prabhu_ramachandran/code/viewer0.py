from traits.api import HasTraits, Instance
from traitsui.api import View, Item, Group
from mayavi.core.ui.api import MlabSceneModel, SceneEditor, MayaviScene
from mayavi.core.api import PipelineBase


class PySPHViewer(HasTraits):

    scene = Instance(MlabSceneModel, ())

    plot = Instance(PipelineBase)

    view = View(
        Group(Item(
            'scene',
            editor=SceneEditor(scene_class=MayaviScene),
            height=400,
            width=400,
            show_label=False
        )),
        resizable=True,
        title='PySPH viewer'
    )

viewer = PySPHViewer()
viewer.configure_traits()
