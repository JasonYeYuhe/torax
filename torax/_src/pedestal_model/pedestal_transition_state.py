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

"""State for persisting intra-step pedestal model outputs."""

import dataclasses
from typing import Self
import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import state
from torax._src.geometry import geometry
from torax._src.internal_boundary_conditions import internal_boundary_conditions as internal_boundary_conditions_lib
from torax._src.pedestal_model import pedestal_model_output as pedestal_model_output_lib
from torax._src.pedestal_model import pedestal_state as pedestal_state_lib
from torax._src.pedestal_model import runtime_params as runtime_params_lib

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PedestalTransitionState:
  """Contains the intra-step pedestal model output.

  Attributes:
    pedestal_model_output: The current pedestal model output for this timestep.
  """

  pedestal_model_output: pedestal_model_output_lib.PedestalStructure

  @classmethod
  def empty(cls) -> Self:
    """Returns a transition state with empty pedestal outputs."""
    return cls(
        pedestal_model_output=(
            pedestal_model_output_lib.PedestalStructure.no_pedestal()
        ),
    )

  def _compute_ramp_fraction(
      self,
      t: array_typing.FloatScalar,
      transition_time_width: array_typing.FloatScalar,
      pedestal_state: pedestal_state_lib.PedestalState,
  ) -> array_typing.FloatScalar:
    """Computes the ramp fraction for a pedestal transition.

    Returns a value in [0, 1] representing the progress of the current
    transition. 0 means the transition just started, 1 means it is complete.

    Args:
      t: Current simulation time (i.e. t + dt when called from the solver).
      transition_time_width: Duration of the transition ramp.
      pedestal_state: Pedestal state carried over from the previous timestep.

    Returns:
      Ramp fraction clipped to [0, 1].
    """
    elapsed = t - pedestal_state.transition_start_time
    safe_width = jnp.where(
        transition_time_width == 0.0, 1.0, transition_time_width
    )
    fraction = jnp.where(
        transition_time_width == 0.0,
        1.0,
        elapsed / safe_width,
    )
    return jnp.clip(fraction, 0.0, 1.0)

  def _apply_transition_ramp_scaling(
      self,
      ramp_fraction: array_typing.FloatScalar,
      pedestal_state: pedestal_state_lib.PedestalState,
  ) -> pedestal_model_output_lib.PedestalStructure:
    """Applies ramp scaling to pedestal model output during transitions.

    During an L-H transition, linearly ramps from L-mode values to the H-mode
    targets. During an H-L transition, ramps from the H-mode targets back to
    the L-mode values.

    The L-mode values are stored in pedestal_state (captured at the start of an
    L->H transition). The H-mode targets are the full pedestal model output.

    Args:
      ramp_fraction: Progress of the current transition, in [0, 1].
      pedestal_state: Pedestal state carried over from the previous timestep.

    Returns:
      Scaled pedestal model output.
    """

    def _interpolate_transition(l_val, h_val):
      """Interpolates between L-mode and H-mode values based on mode."""
      l_to_h_ramp = l_val + ramp_fraction * (h_val - l_val)
      h_to_l_ramp = h_val + ramp_fraction * (l_val - h_val)
      confinement_mode = pedestal_state.confinement_mode
      return jnp.select(
          [
              confinement_mode == runtime_params_lib.ConfinementMode.L_MODE,
              confinement_mode == runtime_params_lib.ConfinementMode.H_MODE,
              confinement_mode
              == runtime_params_lib.ConfinementMode.TRANSITIONING_TO_H_MODE,
              confinement_mode
              == runtime_params_lib.ConfinementMode.TRANSITIONING_TO_L_MODE,
          ],
          [l_val, h_val, l_to_h_ramp, h_to_l_ramp],
      )

    pedestal_model_output = self.pedestal_model_output

    scaled_T_i = _interpolate_transition(
        l_val=pedestal_state.T_i_ped_L_mode,
        h_val=pedestal_model_output.T_i_ped,
    )
    scaled_T_e = _interpolate_transition(
        l_val=pedestal_state.T_e_ped_L_mode,
        h_val=pedestal_model_output.T_e_ped,
    )
    scaled_n_e = _interpolate_transition(
        l_val=pedestal_state.n_e_ped_L_mode,
        h_val=pedestal_model_output.n_e_ped,
    )

    return dataclasses.replace(
        pedestal_model_output,
        T_i_ped=scaled_T_i,
        T_e_ped=scaled_T_e,
        n_e_ped=scaled_n_e,
    )

  def _get_scaled_pedestal_model_output(
      self,
      t: array_typing.FloatScalar,
      transition_time_width: array_typing.FloatScalar,
      pedestal_state: pedestal_state_lib.PedestalState,
  ) -> pedestal_model_output_lib.PedestalStructure:
    """Returns the pedestal model output, scaled by the ramp fraction."""
    ramp_fraction = self._compute_ramp_fraction(
        t=t,
        transition_time_width=transition_time_width,
        pedestal_state=pedestal_state,
    )
    return self._apply_transition_ramp_scaling(
        ramp_fraction=ramp_fraction,
        pedestal_state=pedestal_state,
    )

  def is_ibc_active(
      self,
      pedestal_runtime_params: runtime_params_lib.RuntimeParams,
      pedestal_state: pedestal_state_lib.PedestalState,
  ) -> array_typing.BoolScalar:
    """Returns whether the pedestal IBC is actively controlling the edge."""
    if (
        pedestal_runtime_params.mode
        != runtime_params_lib.Mode.INTERNAL_BOUNDARY_CONDITION
    ):
      return False

    return (
        pedestal_state.confinement_mode
        != runtime_params_lib.ConfinementMode.L_MODE
    )

  def to_internal_boundary_conditions(
      self,
      t: array_typing.FloatScalar,
      pedestal_runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_state: pedestal_state_lib.PedestalState,
  ) -> internal_boundary_conditions_lib.InternalBoundaryConditions:
    """Builds the internal boundary conditions produced by the pedestal model."""
    if (
        pedestal_runtime_params.mode
        == runtime_params_lib.Mode.INTERNAL_BOUNDARY_CONDITION
    ):
      pedestal_model_output = self._get_scaled_pedestal_model_output(
          t=t,
          transition_time_width=pedestal_runtime_params.transition_time_width,
          pedestal_state=pedestal_state,
      )
      return pedestal_model_output.to_internal_boundary_conditions(
          geo,
          core_profiles=core_profiles,
          pedestal_profile_form=pedestal_runtime_params.structure.pedestal_profile_form,
      )
    else:
      # In ADAPTIVE_TRANSPORT mode, transport coefficients govern the edge;
      # edge internal boundary conditions are disabled.
      return internal_boundary_conditions_lib.InternalBoundaryConditions.empty(
          geo
      )
