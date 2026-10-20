# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for ExplicitModelOutputs."""

from absl.testing import absltest
import chex
import jax
from torax._src import explicit_model_outputs as explicit_model_outputs_lib
from torax._src.geometry import circular_geometry
from torax._src.neoclassical.bootstrap_current import (
    base as bootstrap_current_base,
)
from torax._src.pedestal_model import (
    pedestal_transition_state as pedestal_transition_state_lib,
)
from torax._src.sources import source_profiles as source_profiles_lib


class ExplicitModelOutputsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.geo = circular_geometry.CircularConfig().build_geometry()
    self.sources = source_profiles_lib.SourceProfiles(
        bootstrap_current=bootstrap_current_base.BootstrapCurrent.zeros(
            self.geo
        ),
        qei=source_profiles_lib.QeiInfo.zeros(self.geo),
    )
    self.pedestal = (
        pedestal_transition_state_lib.PedestalTransitionState.empty()
    )

  def test_pytree_flatten_and_unflatten(self):
    outputs = explicit_model_outputs_lib.ExplicitModelOutputs(
        sources=self.sources,
        pedestal=self.pedestal,
    )

    leaves, treedef = jax.tree_util.tree_flatten(outputs)
    reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
    chex.assert_trees_all_close(outputs, reconstructed)

  def test_jit_identity(self):
    outputs = explicit_model_outputs_lib.ExplicitModelOutputs(
        sources=self.sources,
        pedestal=self.pedestal,
    )

    @jax.jit
    def identity(x):
      return x

    jitted_outputs = identity(outputs)
    chex.assert_trees_all_close(outputs, jitted_outputs)


if __name__ == '__main__':
  absltest.main()
