import sys
import os

from traits.api import (HasTraits, Instance, Str, List)
from traitsui.api import View, Item, Group, EnumEditor
from mayavi.core.ui.api import MlabSceneModel, SceneEditor, MayaviScene
from mayavi.core.api import PipelineBase

from pysph.base.particle_array import ParticleArray
from pysph.solver.utils import load


class PySPHViewer(HasTraits):

    scene = Instance(MlabSceneModel, ())

    scalar = Str("rho")

    scalar_list = List(Str)

    particle_array = Instance(ParticleArray)

    plot = Instance(PipelineBase)

    file_name = Str

    view = View(
        Group(Item(
            'scene',
            editor=SceneEditor(scene_class=MayaviScene),
            height=400,
            width=400,
            show_label=False
        )),
        Item('file_name'),
        Item('scalar', editor=EnumEditor(name='scalar_list')),
        resizable=True,
        title='PySPH viewer'
    )

    def update_plot(self):
        mlab = self.scene.mlab
        pa = self.particle_array
        if self.plot is None:
            self.plot = mlab.points3d(
                pa.x, pa.y, pa.z,
                getattr(pa, self.scalar),
                mode='point'
            )
            self.plot.actor.property.point_size = 3
        else:
            self.plot.mlab_source.reset(
                x=pa.x, y=pa.y, z=pa.z,
                scalars=getattr(pa, self.scalar)
            )

    def _particle_array_changed(self, pa):
        self.scalar_list = list(pa.properties.keys())
        self.update_plot()

    def _file_name_changed(self, fname):
        if os.path.exists(fname):
            data = load(fname)
            self.particle_array = data['arrays']['fluid']

    def _scalar_changed(self, scalar):
        pa = self.particle_array
        if pa is not None and scalar in pa.properties:
            self.update_plot()


viewer = PySPHViewer(file_name=sys.argv[1])
viewer.configure_traits()
