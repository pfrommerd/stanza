import jax
import jax.numpy as jnp

import mujoco
from mujoco import mjx
import robosuite

import dataclasses
from functools import partial, cached_property

from stanza.dataclasses import dataclass, field
from stanza.env import (
    EnvWrapper, RenderConfig, 
    ImageRender, SequenceRender,
    HtmlRender, Environment,
    EnvironmentRegistry
)
from stanza.policy import Policy, PolicyInput
from stanza.policy.transforms import Transform, chain_transforms
from stanza.env.mujoco import MujocoEnvironment, MujocoState
from stanza import canvas
from jax.random import PRNGKey

XML = """
<mujoco model="base">
	<compiler angle="radian" meshdir="meshes/" inertiagrouprange="0 0" autolimits="true" />
	<size nconmax="5000" njmax="5000" />
	<asset>
		<texture builtin="gradient" height="256" rgb1=".9 .9 1." rgb2=".2 .3 .4" type="skybox" width="256" />
		<texture file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/arenas/../textures/light-gray-floor-tile.png" type="2d" name="texplane" />
		<material name="floorplane" reflectance="0.01" shininess="0.0" specular="0.0" texrepeat="2 2" texture="texplane" texuniform="true" />
		<texture file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/arenas/../textures/ceramic.png" type="cube" name="tex-ceramic" />
		<material name="table_ceramic" reflectance="0.0" shininess="0.0" specular="0.2" texrepeat="1 1" texture="tex-ceramic" />
		<texture file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/arenas/../textures/steel-brushed.png" type="cube" name="tex-steel-brushed" />
		<material name="table_legs_metal" reflectance="0.8" shininess="0.8" texrepeat="1 1" texture="tex-steel-brushed" />
		<texture file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/arenas/../textures/light-gray-plaster.png" type="2d" name="tex-light-gray-plaster" />
		<material name="walls_mat" reflectance="0.0" shininess="0.1" specular="0.1" texrepeat="3 3" texture="tex-light-gray-plaster" texuniform="true" />
		<texture name="textable" builtin="flat" height="512" width="512" rgb1="0.5 0.5 0.5" rgb2="0.5 0.5 0.5" />
		<material name="table_mat" texture="textable" />
		<mesh name="robot0_link0" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/meshes/link0.stl" />
		<mesh name="robot0_link1" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/meshes/link1.stl" />
		<mesh name="robot0_link2" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/meshes/link2.stl" />
		<mesh name="robot0_link3" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/meshes/link3.stl" />
		<mesh name="robot0_link4" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/meshes/link4.stl" />
		<mesh name="robot0_link5" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/meshes/link5.stl" />
		<mesh name="robot0_link6" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/meshes/link6.stl" />
		<mesh name="robot0_link7" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/meshes/link7.stl" />
		<material name="robot0_Face636_001" specular="0.5" shininess="0.45" rgba="0.901961 0.921569 0.929412 1.000000" />
		<material name="robot0_Part__Feature017_001" specular="0.5" shininess="0.45" rgba="1.000000 1.000000 1.000000 1.000000" />
		<material name="robot0_Part__Feature018_001" specular="0.5" shininess="0.45" rgba="1.000000 1.000000 1.000000 1.000000" />
		<material name="robot0_Part__Feature019_001" specular="0.5" shininess="0.45" rgba="1.000000 1.000000 1.000000 1.000000" />
		<material name="robot0_Part__Feature022_001" specular="0.5" shininess="0.45" rgba="0.901961 0.921569 0.929412 1.000000" />
		<material name="robot0_Part__Feature023_001" specular="0.5" shininess="0.45" rgba="0.250980 0.250980 0.250980 1.000000" />
		<material name="robot0_Shell001_001" specular="0.5" shininess="0.45" rgba="0.250980 0.250980 0.250980 1.000000" />
		<material name="robot0_Shell002_001" specular="0.5" shininess="0.45" rgba="0.901961 0.921569 0.929412 1.000000" />
		<material name="robot0_Shell003_001" specular="0.5" shininess="0.45" rgba="0.901961 0.921569 0.929412 1.000000" />
		<material name="robot0_Shell009_001" specular="0.5" shininess="0.45" rgba="0.250980 0.250980 0.250980 1.000000" />
		<material name="robot0_Shell010_001" specular="0.5" shininess="0.45" rgba="0.901961 0.921569 0.929412 1.000000" />
		<material name="robot0_Shell_001" specular="0.5" shininess="0.45" rgba="0.250980 0.250980 0.250980 1.000000" />
		<mesh name="robot0_link0_vis_0" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link0_vis/link0_vis_0.obj" />
		<mesh name="robot0_link0_vis_1" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link0_vis/link0_vis_1.obj" />
		<mesh name="robot0_link0_vis_2" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link0_vis/link0_vis_2.obj" />
		<mesh name="robot0_link0_vis_3" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link0_vis/link0_vis_3.obj" />
		<mesh name="robot0_link0_vis_4" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link0_vis/link0_vis_4.obj" />
		<mesh name="robot0_link0_vis_5" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link0_vis/link0_vis_5.obj" />
		<mesh name="robot0_link0_vis_6" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link0_vis/link0_vis_6.obj" />
		<mesh name="robot0_link0_vis_7" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link0_vis/link0_vis_7.obj" />
		<mesh name="robot0_link0_vis_8" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link0_vis/link0_vis_8.obj" />
		<mesh name="robot0_link0_vis_9" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link0_vis/link0_vis_9.obj" />
		<mesh name="robot0_link0_vis_10" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link0_vis/link0_vis_10.obj" />
		<mesh name="robot0_link0_vis_11" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link0_vis/link0_vis_11.obj" />
		<material name="robot0_Part__Feature_001" specular="0.5" shininess="0.45" rgba="1.000000 1.000000 1.000000 1.000000" />
		<mesh name="robot0_link1_vis" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link1_vis/link1_vis.obj" />
		<material name="robot0_Part__Feature024" specular="0.5" shininess="0.45" rgba="1.000000 1.000000 1.000000 1.000000" />
		<mesh name="robot0_link2_vis" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link2_vis/link2_vis.obj" />
		<material name="robot0_Part__Feature001_010_001_002" specular="0.5" shininess="0.45" rgba="1.000000 1.000000 1.000000 1.000000" />
		<material name="robot0_Part__Feature002_007_001_002" specular="0.5" shininess="0.45" rgba="1.000000 1.000000 1.000000 1.000000" />
		<material name="robot0_Part__Feature003_004_001_002" specular="0.5" shininess="0.45" rgba="1.000000 1.000000 1.000000 1.000000" />
		<material name="robot0_Part__Feature_001_001_001_002" specular="0.5" shininess="0.45" rgba="0.250980 0.250980 0.250980 1.000000" />
		<mesh name="robot0_link3_vis_0" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link3_vis/link3_vis_0.obj" />
		<mesh name="robot0_link3_vis_1" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link3_vis/link3_vis_1.obj" />
		<mesh name="robot0_link3_vis_2" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link3_vis/link3_vis_2.obj" />
		<mesh name="robot0_link3_vis_3" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link3_vis/link3_vis_3.obj" />
		<material name="robot0_Part__Feature001_001_003_001" specular="0.5" shininess="0.45" rgba="1.000000 1.000000 1.000000 1.000000" />
		<material name="robot0_Part__Feature002_001_003_001" specular="0.5" shininess="0.45" rgba="0.250980 0.250980 0.250980 1.000000" />
		<material name="robot0_Part__Feature003_001_003_001" specular="0.5" shininess="0.45" rgba="1.000000 1.000000 1.000000 1.000000" />
		<material name="robot0_Part__Feature_002_003_001" specular="0.5" shininess="0.45" rgba="1.000000 1.000000 1.000000 1.000000" />
		<mesh name="robot0_link4_vis_0" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link4_vis/link4_vis_0.obj" />
		<mesh name="robot0_link4_vis_1" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link4_vis/link4_vis_1.obj" />
		<mesh name="robot0_link4_vis_2" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link4_vis/link4_vis_2.obj" />
		<mesh name="robot0_link4_vis_3" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link4_vis/link4_vis_3.obj" />
		<material name="robot0_Part__Feature_002_004_003" specular="0.5" shininess="0.45" rgba="1.000000 1.000000 1.000000 1.000000" />
		<material name="robot0_Shell001_001_001_003" specular="0.5" shininess="0.45" rgba="0.250000 0.250000 0.250000 1.000000" />
		<material name="robot0_Shell_001_001_003" specular="0.5" shininess="0.45" rgba="1.000000 1.000000 1.000000 1.000000" />
		<mesh name="robot0_link5_vis_0" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link5_vis/link5_vis_0.obj" />
		<mesh name="robot0_link5_vis_1" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link5_vis/link5_vis_1.obj" />
		<mesh name="robot0_link5_vis_2" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link5_vis/link5_vis_2.obj" />
		<material name="robot0_Face064_002_001_002_001" specular="0.5" shininess="0.45" rgba="1.000000 0.000000 0.000000 1.000000" />
		<material name="robot0_Face065_002_001_002_001" specular="0.5" shininess="0.45" rgba="0.000000 1.000000 0.000000 1.000000" />
		<material name="robot0_Face374_002_001_002_001" specular="0.5" shininess="0.45" rgba="1.000000 1.000000 1.000000 1.000000" />
		<material name="robot0_Face539_002_001_002_001" specular="0.5" shininess="0.45" rgba="0.250980 0.250980 0.250980 1.000000" />
		<material name="robot0_Part__Feature001_009_001_002_001" specular="0.5" shininess="0.45" rgba="0.250980 0.250980 0.250980 1.000000" />
		<material name="robot0_Part__Feature002_006_001_002_001" specular="0.5" shininess="0.45" rgba="0.250980 0.250980 0.250980 1.000000" />
		<material name="robot0_Shell002_002_001_002_001" specular="0.5" shininess="0.45" rgba="1.000000 1.000000 1.000000 1.000000" />
		<material name="robot0_Shell003_002_001_002_001" specular="0.5" shininess="0.45" rgba="1.000000 1.000000 1.000000 1.000000" />
		<material name="robot0_Shell004_001_001_002_001" specular="0.5" shininess="0.45" rgba="1.000000 1.000000 1.000000 1.000000" />
		<material name="robot0_Shell005_001_001_002_001" specular="0.5" shininess="0.45" rgba="1.000000 1.000000 1.000000 1.000000" />
		<material name="robot0_Shell006_003_002_001" specular="0.5" shininess="0.45" rgba="0.901961 0.921569 0.929412 1.000000" />
		<material name="robot0_Shell007_002_002_001" specular="0.5" shininess="0.45" rgba="0.250000 0.250000 0.250000 1.000000" />
		<material name="robot0_Shell011_002_002_001" specular="0.5" shininess="0.45" rgba="1.000000 1.000000 1.000000 1.000000" />
		<material name="robot0_Shell012_002_002_001" specular="0.5" shininess="0.45" rgba="1.000000 1.000000 1.000000 1.000000" />
		<material name="robot0_Shell_003_001_002_001" specular="0.5" shininess="0.45" rgba="0.250980 0.250980 0.250980 1.000000" />
		<material name="robot0_Union001_001_001_002_001" specular="0.5" shininess="0.45" rgba="0.039216 0.541176 0.780392 1.000000" />
		<material name="robot0_Union_001_001_002_001" specular="0.5" shininess="0.45" rgba="0.039216 0.541176 0.780392 1.000000" />
		<mesh name="robot0_link6_vis_0" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link6_vis/link6_vis_0.obj" />
		<mesh name="robot0_link6_vis_1" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link6_vis/link6_vis_1.obj" />
		<mesh name="robot0_link6_vis_2" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link6_vis/link6_vis_2.obj" />
		<mesh name="robot0_link6_vis_3" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link6_vis/link6_vis_3.obj" />
		<mesh name="robot0_link6_vis_4" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link6_vis/link6_vis_4.obj" />
		<mesh name="robot0_link6_vis_5" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link6_vis/link6_vis_5.obj" />
		<mesh name="robot0_link6_vis_6" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link6_vis/link6_vis_6.obj" />
		<mesh name="robot0_link6_vis_7" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link6_vis/link6_vis_7.obj" />
		<mesh name="robot0_link6_vis_8" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link6_vis/link6_vis_8.obj" />
		<mesh name="robot0_link6_vis_9" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link6_vis/link6_vis_9.obj" />
		<mesh name="robot0_link6_vis_10" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link6_vis/link6_vis_10.obj" />
		<mesh name="robot0_link6_vis_11" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link6_vis/link6_vis_11.obj" />
		<mesh name="robot0_link6_vis_12" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link6_vis/link6_vis_12.obj" />
		<mesh name="robot0_link6_vis_13" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link6_vis/link6_vis_13.obj" />
		<mesh name="robot0_link6_vis_14" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link6_vis/link6_vis_14.obj" />
		<mesh name="robot0_link6_vis_15" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link6_vis/link6_vis_15.obj" />
		<mesh name="robot0_link6_vis_16" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link6_vis/link6_vis_16.obj" />
		<material name="robot0_Part__Mirroring001_004_002" specular="0.5" shininess="0.45" rgba="0.250980 0.250980 0.250980 1.000000" />
		<material name="robot0_Part__Mirroring002_004_001" specular="0.5" shininess="0.45" rgba="0.250980 0.250980 0.250980 1.000000" />
		<material name="robot0_Part__Mirroring003_004_001" specular="0.5" shininess="0.45" rgba="0.250980 0.250980 0.250980 1.000000" />
		<material name="robot0_Part__Mirroring004_004_002" specular="0.5" shininess="0.45" rgba="1.000000 1.000000 1.000000 1.000000" />
		<material name="robot0_Part__Mirroring005_004_001" specular="0.5" shininess="0.45" rgba="0.250980 0.250980 0.250980 1.000000" />
		<material name="robot0_Part__Mirroring006_004_001" specular="0.5" shininess="0.45" rgba="0.250980 0.250980 0.250980 1.000000" />
		<material name="robot0_Part__Mirroring007_004_001" specular="0.5" shininess="0.45" rgba="0.250980 0.250980 0.250980 1.000000" />
		<material name="robot0_Part__Mirroring_004_001" specular="0.5" shininess="0.45" rgba="0.898039 0.917647 0.929412 1.000000" />
		<mesh name="robot0_link7_vis_0" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link7_vis/link7_vis_0.obj" />
		<mesh name="robot0_link7_vis_1" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link7_vis/link7_vis_1.obj" />
		<mesh name="robot0_link7_vis_2" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link7_vis/link7_vis_2.obj" />
		<mesh name="robot0_link7_vis_3" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link7_vis/link7_vis_3.obj" />
		<mesh name="robot0_link7_vis_4" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link7_vis/link7_vis_4.obj" />
		<mesh name="robot0_link7_vis_5" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link7_vis/link7_vis_5.obj" />
		<mesh name="robot0_link7_vis_6" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link7_vis/link7_vis_6.obj" />
		<mesh name="robot0_link7_vis_7" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link7_vis/link7_vis_7.obj" />
		<mesh name="mount0_pedestal" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/mounts/meshes/rethink_mount/pedestal.stl" />
		<mesh name="gripper0_hand" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/grippers/meshes/panda_gripper/hand.stl" />
		<mesh name="gripper0_hand_vis" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/grippers/meshes/panda_gripper/hand_vis.stl" />
		<mesh name="gripper0_finger" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/grippers/meshes/panda_gripper/finger.stl" />
		<mesh name="gripper0_finger_vis" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/grippers/meshes/panda_gripper/finger_vis.stl" />
		<mesh name="gripper0_finger_vis2" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/grippers/meshes/panda_gripper/finger_longer.stl" />
		<texture type="cube" name="cube_redwood" file="/Users/chryu/stanza/stanza/.venv/lib/python3.12/site-packages/robosuite/models/assets/textures/red-wood.png" />
		<material texrepeat="1 1" specular="0.4" shininess="0.1" name="cube_redwood_mat" texture="cube_redwood" />
	</asset>
	<visual>
		<map znear="0.001" />
	</visual>
	<actuator>
		<motor ctrllimited="true" ctrlrange="-80.0 80.0" joint="robot0_joint1" name="robot0_torq_j1" />
		<motor ctrllimited="true" ctrlrange="-80.0 80.0" joint="robot0_joint2" name="robot0_torq_j2" />
		<motor ctrllimited="true" ctrlrange="-80.0 80.0" joint="robot0_joint3" name="robot0_torq_j3" />
		<motor ctrllimited="true" ctrlrange="-80.0 80.0" joint="robot0_joint4" name="robot0_torq_j4" />
		<motor ctrllimited="true" ctrlrange="-80.0 80.0" joint="robot0_joint5" name="robot0_torq_j5" />
		<motor ctrllimited="true" ctrlrange="-12.0 12.0" joint="robot0_joint6" name="robot0_torq_j6" />
		<motor ctrllimited="true" ctrlrange="-12.0 12.0" joint="robot0_joint7" name="robot0_torq_j7" />
		<position ctrllimited="true" ctrlrange="0.0 0.04" joint="gripper0_finger_joint1" kp="1000" name="gripper0_gripper_finger_joint1" forcelimited="true" forcerange="-20 20" />
		<position ctrllimited="true" ctrlrange="-0.04 0.0" joint="gripper0_finger_joint2" kp="1000" name="gripper0_gripper_finger_joint2" forcelimited="true" forcerange="-20 20" />
	</actuator>
	<worldbody>
		<geom condim="3" group="1" material="floorplane" name="floor" pos="0.0 0.0 0.0" size="3 3 .125" type="plane" />
		<geom pos="-1.25 2.25 1.5" quat="0.6532815 0.6532815 0.2705981 0.2705981" size="1.06 1.5 0.01" type="box" conaffinity="0" contype="0" group="1" name="wall_leftcorner_visual" material="walls_mat" />
		<geom pos="-1.25 -2.25 1.5" quat="0.6532815 0.6532815 -0.2705981 -0.2705981" size="1.06 1.5 0.01" type="box" conaffinity="0" contype="0" group="1" name="wall_rightcorner_visual" material="walls_mat" />
		<geom pos="1.25 3.0 1.5" quat="0.7071 0.7071 0 0" size="1.75 1.5 0.01" type="box" conaffinity="0" contype="0" group="1" name="wall_left_visual" material="walls_mat" />
		<geom pos="1.25 -3.0 1.5" quat="0.7071 -0.7071 0 0" size="1.75 1.5 0.01" type="box" conaffinity="0" contype="0" group="1" name="wall_right_visual" material="walls_mat" />
		<geom pos="-2.0 0.0 1.5" quat="0.5 0.5 0.5 0.5" size="1.5 1.5 0.01" type="box" conaffinity="0" contype="0" group="1" name="wall_rear_visual" material="walls_mat" />
		<geom pos="3.0 0.0 1.5" quat="0.5 0.5 -0.5 -0.5" size="3 1.5 0.01" type="box" conaffinity="0" contype="0" group="1" name="wall_front_visual" material="walls_mat" />
		<body name="table" pos="0.0 0.0 0.775">
			<geom pos="0 0 0" size="0.4 0.4 0.025" type="box" group="0" name="table_collision" friction="1.0 0.005 0.0001" rgba="0.5 0.5 0 1" />
			<geom pos="0 0 0" size="0.4 0.4 0.025" type="box" conaffinity="0" contype="0" group="1" name="table_visual" material="table_ceramic" />
			<site pos="0.0 0.0 0.025" name="table_top" size="0.001 0.001 0.001" rgba="0 0 0 0" />
		</body>
		<light pos="1.0 1.0 1.5" dir="-0.2 -0.2 -1" specular="0.3 0.3 0.3" directional="true" castshadow="false" />
		<camera mode="fixed" name="frontview" pos="1.6 0.0 1.45" quat="0.56 0.43 0.43 0.56" />
		<camera mode="fixed" name="birdview" pos="-0.2 0.0 3.0" quat="0.7071 0 0 0.7071" />
		<camera mode="fixed" name="agentview" pos="0.5 0.0 1.35" quat="0.653 0.271 0.271 0.653" />
		<camera mode="fixed" name="sideview" pos="-0.05651774593317116 1.2761224129427358 1.4879572214102434" quat="0.009905065491771751 0.006877963156909582 0.5912228352893879 0.806418094001364" />
		<body name="robot0_base" pos="-0.56 0.0 0.912">
			<camera mode="fixed" name="robot0_robotview" pos="1.0 0 0.4" quat="0.653 0.271 0.271 0.653" />
			<inertial diaginertia="0 0 0" mass="0" pos="0 0 0" />
			<body name="robot0_link0" pos="0 0 0">
				<inertial pos="0 0 0.05" mass="4" diaginertia="0.4 0.4 0.4" />
				<geom mesh="robot0_link0_vis_0" material="robot0_Shell010_001" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g0_vis" />
				<geom mesh="robot0_link0_vis_1" material="robot0_Shell009_001" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g1_vis" />
				<geom mesh="robot0_link0_vis_2" material="robot0_Shell003_001" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g2_vis" />
				<geom mesh="robot0_link0_vis_3" material="robot0_Shell_001" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g3_vis" />
				<geom mesh="robot0_link0_vis_4" material="robot0_Shell002_001" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g4_vis" />
				<geom mesh="robot0_link0_vis_5" material="robot0_Shell001_001" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g5_vis" />
				<geom mesh="robot0_link0_vis_6" material="robot0_Face636_001" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g6_vis" />
				<geom mesh="robot0_link0_vis_7" material="robot0_Part__Feature018_001" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g7_vis" />
				<geom mesh="robot0_link0_vis_8" material="robot0_Part__Feature019_001" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g8_vis" />
				<geom mesh="robot0_link0_vis_9" material="robot0_Part__Feature023_001" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g9_vis" />
				<geom mesh="robot0_link0_vis_10" material="robot0_Part__Feature022_001" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g10_vis" />
				<geom mesh="robot0_link0_vis_11" material="robot0_Part__Feature017_001" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g11_vis" />
				<geom type="mesh" group="0" mesh="robot0_link0" name="robot0_link0_collision" rgba="0 0.5 0 1" />
				<body name="robot0_link1" pos="0 0 0.333">
					<inertial pos="0 0 -0.07" mass="3" diaginertia="0.3 0.3 0.3" />
					<joint name="robot0_joint1" pos="0 0 0" axis="0 0 1" limited="true" range="-2.8973 2.8973" damping="0.1" armature="5.0" />
					<geom material="robot0_Part__Feature_001" mesh="robot0_link1_vis" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g12_vis" />
					<geom type="mesh" group="0" mesh="robot0_link1" name="robot0_link1_collision" rgba="0 0.5 0 1" />
					<body name="robot0_link2" pos="0 0 0" quat="0.707107 -0.707107 0 0">
						<inertial pos="0 -0.1 0" mass="3" diaginertia="0.3 0.3 0.3" />
						<joint name="robot0_joint2" pos="0 0 0" axis="0 0 1" limited="true" range="-1.7628 1.7628" damping="0.1" armature="2.5" />
						<geom material="robot0_Part__Feature024" mesh="robot0_link2_vis" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g13_vis" />
						<geom type="mesh" group="0" mesh="robot0_link2" name="robot0_link2_collision" rgba="0 0.5 0 1" />
						<body name="robot0_link3" pos="0 -0.316 0" quat="0.707107 0.707107 0 0">
							<inertial pos="0.04 0 -0.05" mass="2" diaginertia="0.2 0.2 0.2" />
							<joint name="robot0_joint3" pos="0 0 0" axis="0 0 1" limited="true" range="-2.8973 2.8973" damping="0.1" armature="1.6666666666666667" />
							<geom mesh="robot0_link3_vis_0" material="robot0_Part__Feature003_004_001_002" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g14_vis" />
							<geom mesh="robot0_link3_vis_1" material="robot0_Part__Feature002_007_001_002" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g15_vis" />
							<geom mesh="robot0_link3_vis_2" material="robot0_Part__Feature001_010_001_002" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g16_vis" />
							<geom mesh="robot0_link3_vis_3" material="robot0_Part__Feature_001_001_001_002" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g17_vis" />
							<geom type="mesh" group="0" mesh="robot0_link3" name="robot0_link3_collision" rgba="0 0.5 0 1" />
							<body name="robot0_link4" pos="0.0825 0 0" quat="0.707107 0.707107 0 0">
								<inertial pos="-0.04 0.05 0" mass="2" diaginertia="0.2 0.2 0.2" />
								<joint name="robot0_joint4" pos="0 0 0" axis="0 0 1" limited="true" range="-3.0718 -0.0698" damping="0.1" armature="1.25" />
								<geom mesh="robot0_link4_vis_0" material="robot0_Part__Feature001_001_003_001" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g18_vis" />
								<geom mesh="robot0_link4_vis_1" material="robot0_Part__Feature003_001_003_001" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g19_vis" />
								<geom mesh="robot0_link4_vis_2" material="robot0_Part__Feature002_001_003_001" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g20_vis" />
								<geom mesh="robot0_link4_vis_3" material="robot0_Part__Feature_002_003_001" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g21_vis" />
								<geom type="mesh" group="0" mesh="robot0_link4" name="robot0_link4_collision" rgba="0 0.5 0 1" />
								<body name="robot0_link5" pos="-0.0825 0.384 0" quat="0.707107 -0.707107 0 0">
									<inertial pos="0 0 -0.15" mass="2" diaginertia="0.2 0.2 0.2" />
									<joint name="robot0_joint5" pos="0 0 0" axis="0 0 1" limited="true" range="-2.8973 2.8973" damping="0.1" armature="1.0" />
									<geom mesh="robot0_link5_vis_0" material="robot0_Shell001_001_001_003" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g22_vis" />
									<geom mesh="robot0_link5_vis_1" material="robot0_Shell_001_001_003" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g23_vis" />
									<geom mesh="robot0_link5_vis_2" material="robot0_Part__Feature_002_004_003" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g24_vis" />
									<geom type="mesh" group="0" mesh="robot0_link5" name="robot0_link5_collision" rgba="0 0.5 0 1" />
									<body name="robot0_link6" pos="0 0 0" quat="0.707107 0.707107 0 0">
										<inertial pos="0.06 0 0" mass="1.5" diaginertia="0.1 0.1 0.1" />
										<joint name="robot0_joint6" pos="0 0 0" axis="0 0 1" limited="true" range="-0.0175 3.7525" damping="0.01" armature="0.8333333333333334" />
										<geom mesh="robot0_link6_vis_0" material="robot0_Shell006_003_002_001" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g25_vis" />
										<geom mesh="robot0_link6_vis_1" material="robot0_Shell011_002_002_001" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g26_vis" />
										<geom mesh="robot0_link6_vis_2" material="robot0_Shell007_002_002_001" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g27_vis" />
										<geom mesh="robot0_link6_vis_3" material="robot0_Shell005_001_001_002_001" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g28_vis" />
										<geom mesh="robot0_link6_vis_4" material="robot0_Shell004_001_001_002_001" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g29_vis" />
										<geom mesh="robot0_link6_vis_5" material="robot0_Shell003_002_001_002_001" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g30_vis" />
										<geom mesh="robot0_link6_vis_6" material="robot0_Shell002_002_001_002_001" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g31_vis" />
										<geom mesh="robot0_link6_vis_7" material="robot0_Union001_001_001_002_001" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g32_vis" />
										<geom mesh="robot0_link6_vis_8" material="robot0_Union_001_001_002_001" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g33_vis" />
										<geom mesh="robot0_link6_vis_9" material="robot0_Face539_002_001_002_001" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g34_vis" />
										<geom mesh="robot0_link6_vis_10" material="robot0_Shell_003_001_002_001" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g35_vis" />
										<geom mesh="robot0_link6_vis_11" material="robot0_Face374_002_001_002_001" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g36_vis" />
										<geom mesh="robot0_link6_vis_12" material="robot0_Face065_002_001_002_001" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g37_vis" />
										<geom mesh="robot0_link6_vis_13" material="robot0_Face064_002_001_002_001" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g38_vis" />
										<geom mesh="robot0_link6_vis_14" material="robot0_Part__Feature002_006_001_002_001" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g39_vis" />
										<geom mesh="robot0_link6_vis_15" material="robot0_Part__Feature001_009_001_002_001" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g40_vis" />
										<geom mesh="robot0_link6_vis_16" material="robot0_Shell012_002_002_001" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g41_vis" />
										<geom type="mesh" group="0" mesh="robot0_link6" name="robot0_link6_collision" rgba="0 0.5 0 1" />
										<body name="robot0_link7" pos="0.088 0 0" quat="0.707107 0.707107 0 0">
											<inertial pos="0 0 0.08" mass="0.5" diaginertia="0.05 0.05 0.05" />
											<joint name="robot0_joint7" pos="0 0 0" axis="0 0 1" limited="true" range="-2.8973 2.8973" damping="0.01" armature="0.7142857142857143" />
											<geom mesh="robot0_link7_vis_0" material="robot0_Part__Mirroring004_004_002" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g42_vis" />
											<geom mesh="robot0_link7_vis_1" material="robot0_Part__Mirroring001_004_002" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g43_vis" />
											<geom mesh="robot0_link7_vis_2" material="robot0_Part__Mirroring007_004_001" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g44_vis" />
											<geom mesh="robot0_link7_vis_3" material="robot0_Part__Mirroring006_004_001" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g45_vis" />
											<geom mesh="robot0_link7_vis_4" material="robot0_Part__Mirroring005_004_001" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g46_vis" />
											<geom mesh="robot0_link7_vis_5" material="robot0_Part__Mirroring003_004_001" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g47_vis" />
											<geom mesh="robot0_link7_vis_6" material="robot0_Part__Mirroring002_004_001" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g48_vis" />
											<geom mesh="robot0_link7_vis_7" material="robot0_Part__Mirroring_004_001" type="mesh" contype="0" conaffinity="0" group="1" name="robot0_g49_vis" />
											<geom type="mesh" group="0" mesh="robot0_link7" name="robot0_link7_collision" rgba="0 0.5 0 1" />
											<body name="robot0_right_hand" pos="0 0 0.1065" quat="0.924 0 0 -0.383">
												<inertial pos="0 0 0" mass="0.5" diaginertia="0.05 0.05 0.05" />
												<camera mode="fixed" name="robot0_eye_in_hand" pos="0.05 0 0" quat="0 0.707108 0.707108 0" fovy="75" />
												<body name="gripper0_right_gripper" pos="0 0 0" quat="0.707107 0 0 -0.707107">
													<site name="gripper0_ft_frame" pos="0 0 0" size="0.01 0.01 0.01" rgba="1 0 0 1" type="sphere" group="1" />
													<inertial pos="0 0 0.17" quat="0.707107 0.707107 0 0" mass="0.3" diaginertia="0.09 0.07 0.05" />
													<geom pos="0 0 0." quat="0.707107 0 0 0.707107" type="mesh" contype="0" conaffinity="0" group="1" mesh="gripper0_hand_vis" name="gripper0_hand_visual" rgba="1 1 1 1" />
													<geom pos="0 0 0." quat="0.707107 0 0 0.707107" type="mesh" mesh="gripper0_hand" group="0" name="gripper0_hand_collision" rgba="0 0 0.5 1" />
													<body name="gripper0_eef" pos="0 0 0.097" quat="1 0 0 0">
														<site name="gripper0_grip_site" pos="0 0 0" size="0.01 0.01 0.01" rgba="1 0 0 0.5" type="sphere" group="1" />
													</body>
													<body name="gripper0_leftfinger" pos="0 0 0.0524" quat="0.707107 0 0 0.707107">
														<inertial pos="0 0 0.05" mass="0.1" diaginertia="0.01 0.01 0.005" />
														<joint name="gripper0_finger_joint1" pos="0 0 0" axis="0 1 0" type="slide" limited="true" range="0.0 0.04" damping="100" armature="1.0" />
														<geom type="mesh" contype="0" conaffinity="0" group="1" mesh="gripper0_finger_vis" name="gripper0_finger1_visual" rgba="0.499 0.499 0.499 1" />
														<geom type="mesh" group="0" conaffinity="1" contype="0" solref="0.02 1" friction="1 0.005 0.0001" condim="4" mesh="gripper0_finger" name="gripper0_finger1_collision" rgba="0 0 0.5 1" />
														<body name="gripper0_finger_joint1_tip" pos="0 0.0085 0.056">
															<inertial pos="0 0 0" quat="0 0 0 1" mass="0.01" diaginertia="0.01 0.01 0.01" />
															<geom size="0.008 0.004 0.008" pos="0 -0.005 -0.015" quat="0 0 0 1" type="box" group="0" solref="0.01 0.5" friction="2 0.05 0.0001" conaffinity="1" contype="1" name="gripper0_finger1_pad_collision" rgba="0 0 0.5 1" />
														</body>
													</body>
													<body name="gripper0_rightfinger" pos="0 0 0.0524" quat="0.707107 0 0 0.707107">
														<inertial pos="0 0 0.05" mass="0.1" diaginertia="0.01 0.01 0.005" />
														<joint name="gripper0_finger_joint2" pos="0 0 0" axis="0 1 0" type="slide" limited="true" range="-0.04 0.0" damping="100" armature="1.0" />
														<geom quat="0 0 0 1" type="mesh" contype="0" conaffinity="0" group="1" mesh="gripper0_finger_vis" name="gripper0_finger2_visual" rgba="0.499 0.499 0.499 1" />
														<geom quat="0 0 0 1" type="mesh" group="0" conaffinity="1" contype="0" solref="0.02 1" friction="1 0.005 0.0001" condim="4" mesh="gripper0_finger" name="gripper0_finger2_collision" rgba="0 0 0.5 1" />
														<body name="gripper0_finger_joint2_tip" pos="0 -0.0085 0.056">
															<inertial pos="0 0 0" quat="0 0 0 1" mass="0.01" diaginertia="0.01 0.01 0.01" />
															<geom size="0.008 0.004 0.008" pos="0 0.005 -0.015" quat="0 0 0 1" type="box" group="0" solref="0.01 0.5" friction="2 0.05 0.0001" conaffinity="1" contype="1" name="gripper0_finger2_pad_collision" rgba="0 0 0.5 1" />
														</body>
													</body>
												</body>
											</body>
										</body>
									</body>
								</body>
							</body>
						</body>
					</body>
				</body>
			</body>
			<body name="mount0_base" pos="0.0 0.0 0.01">
				<body name="mount0_controller_box" pos="0 0 0">
					<inertial diaginertia="1.71363 1.27988 0.809981" mass="46.64" pos="-0.325 0 -0.38" />
					<geom pos="-0.325 0 -0.38" size="0.11 0.2 0.265" type="box" name="mount0_controller_box_col" rgba="0.5 0.5 0 1" />
				</body>
				<body name="mount0_pedestal_feet" pos="0 0 0">
					<inertial diaginertia="8.16095 9.59375 15.0785" mass="167.09" pos="-0.1225 0 -0.758" />
					<geom pos="-0.1225 0 -0.758" size="0.385 0.35 0.155" type="box" name="mount0_pedestal_feet_col" rgba="0.5 0.5 0 1" />
				</body>
				<body name="mount0_torso" pos="0 0 0">
					<inertial diaginertia="1e-08 1e-08 1e-08" mass="0.0001" pos="0 0 0" />
					<geom conaffinity="0" contype="0" group="1" pos="0 0 -0.05" size="0.05 0.05 0.05" type="box" name="mount0_torso_vis" rgba="0.2 0.2 0.2 1" />
				</body>
				<body name="mount0_pedestal" pos="0 0 0">
					<inertial diaginertia="6.0869 5.81635 4.20915" mass="60.864" pos="0 0 0" quat="0.659267 -0.259505 -0.260945 0.655692" />
					<geom conaffinity="0" contype="0" group="1" mesh="mount0_pedestal" type="mesh" name="mount0_pedestal_vis" rgba="0.2 0.2 0.2 1" />
				</body>
			</body>
		</body>
		<body pos="0 0 0" name="cube_main">
			<geom rgba="0.5 0 0 1" density="1000" friction="1 0.005 0.0001" solref="0.02 1.0" solimp="0.9 0.95 0.001" type="box" size="0.02059187440552144 0.02066114428489642 0.020229231326575312" pos="0 0 0" group="0" name="cube_g0" />
			<geom conaffinity="0" contype="0" mass="1e-8" material="cube_redwood_mat" type="box" size="0.02059187440552144 0.02066114428489642 0.020229231326575312" pos="0 0 0" group="1" name="cube_g0_vis" />
			<joint type="free" name="cube_joint0" />
			<site type="sphere" group="0" pos="0 0 0" size="0.002 0.002 0.002" rgba="1 0 0 1" name="cube_default_site" />
		</body>
	</worldbody>
	<equality></equality>
	<sensor>
		<force name="gripper0_force_ee" site="gripper0_ft_frame" />
		<torque name="gripper0_torque_ee" site="gripper0_ft_frame" />
	</sensor>
	<tendon />
	<contact />
</mujoco>
"""


@dataclass
class RobomimicEnv(MujocoEnvironment):
    task: str = field(pytree_node=False)
    dataset_type: str = field(pytree_node=False)
    env_meta: dict = field(pytree_node=False)

    @property
    def xml(self):
        """Return XML string for the environment. """
        with jax.ensure_compile_time_eval():
            # robosuite.make returns a MujocoEnv
            #xml = robosuite.make(self.env_meta["env_name"], **self.env_meta["env_kwargs"]).model.get_xml()
            #print(xml)
            return XML
            
    @jax.jit
    def full_state(self, state: MujocoState):
        mjx_data = mjx.make_data(self.mjx_model)
        mjx_data = mjx_data.replace(qpos=state.q, qvel=state.qd)
        return mjx.forward(self.mjx_model, mjx_data)
    
    @jax.jit
    def sample_action(self, rng_key: jax.Array): ...

    @jax.jit
    def sample_state(self, rng_key):
        return self.reset(rng_key)

    @jax.jit
    def reset(self, rng_key : jax.Array): ...

    
    @jax.jit
    def observe(self, state): ...
    
    @jax.jit
    def render(self, config: RenderConfig, state: MujocoState) -> jax.Array:
        if state.qpos.ndim == 1:
            state = jax.tree_map(lambda x: x[None], state)
        return self.brax_render(self.mj_model, state)