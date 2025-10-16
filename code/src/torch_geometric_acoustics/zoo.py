"""
List of room geometries from benchmark datasets.
The HAA dataset part is essentially identical to their implementation.
"""

from torch_geometric_acoustics.mesh import Mesh


def coupled_rooms_office_to_kitchen(with_door=True):
    mesh = Mesh()
    z = 2.9
    offset_x = -0.59 - 0.4
    offset_y = 8.43

    mesh.add_shoebox_with_coords([offset_x + 0, offset_y + 0, 0], 0.9, -5, z)
    mesh.add_shoebox_with_coords([offset_x + 0.9, offset_y + -0, 0], 2.6, -5, z)
    mesh.add_shoebox_with_coords([offset_x + 0.9, offset_y + -5, 0], 2.6, -2.1, z)
    mesh.add_shoebox_with_coords([offset_x + 0.9, offset_y + -7.1, 0], 2.6, -2, z)
    mesh.add_shoebox_with_coords([offset_x + 0, offset_y + -7.1, 0], 0.9, -2, z)
    mesh.remove_patch_with_vertex_ids((10, 11, 9, 8))
    mesh.remove_patch_with_vertex_ids((11, 15, 13, 9))
    mesh.remove_patch_with_vertex_ids((15, 19, 17, 13))

    mesh.add_shoebox_with_coords([offset_x + 3.5, offset_y + -0, 0], 4.4, -8.8, z)
    mesh.remove_patch_with_vertex_ids((10, 28, 25, 8))
    mesh.add_parallelogram([offset_x + 3.5, offset_y + 0, 0], [0, 0, z], [0, -8.1, 0])

    mesh.add_shoebox_with_coords([offset_x + 3.5, offset_y + -8.8, 0], 4.4, -5.1, z)

    if with_door:
        mesh.add_parallelogram(
            [offset_x + 3.5, offset_y + -8.8, 0], [0, 0, z], [-0.7, 0, 0]
        )

    return mesh


def coupled_rooms_office_to_anechoic_chamber(with_door=True):
    mesh = Mesh()
    chamber_z = 8.2
    office_z = 2.8
    offset_x = -0.32 - 0.9
    offset_y = -1.78 - 2.1
    mesh.add_shoebox_with_coords([offset_x + 0, offset_y + 0, 0], 3.8, 6, office_z)
    mesh.add_shoebox_with_coords(
        [offset_x + 3.8, offset_y + 1.3, 0], 8.2, 8.2, chamber_z
    )
    mesh.remove_patch_with_vertex_ids((7, 5, 1, 4))
    mesh.remove_patch_with_vertex_ids((14, 11, 8, 10))
    mesh.add_parallelogram(
        [offset_x + 3.8, offset_y + 0, 0], [0, 0, office_z], [0, 1.3, 0]
    )
    id = mesh.next_id
    mesh.add_parallelogram(
        [offset_x + 3.8, offset_y + 1.3, office_z],
        [0, 0, chamber_z - office_z],
        [0, 4.7, 0],
        object_id=id,
    )
    mesh.add_parallelogram(
        [offset_x + 3.8, offset_y + 6, office_z],
        [0, 0, chamber_z - office_z],
        [0, 3.5, 0],
        object_id=id,
    )
    mesh.add_parallelogram(
        [offset_x + 3.8, offset_y + 6, 0],
        [0, 0, office_z],
        [0, 3.5, 0],
        object_id=id,
    )
    mesh.add_parallelogram(
        [offset_x + 3.8, offset_y + 1.3, 0],
        [0, 0, office_z],
        [0, 1.8, 0],
    )
    mesh.add_parallelogram(
        [offset_x + 3.8, offset_y + 6, 0],
        [0, 0, office_z],
        [0, -1.5, 0],
    )
    if with_door:
        mesh.add_parallelogram(
            [offset_x + 3.8, offset_y + 3.1, 0],
            [0, 0, office_z],
            [-1.2, 0, 0],
        )
        mesh.add_parallelogram(
            [offset_x + 3.8, offset_y + 3.1, 0],
            [0, 0, office_z],
            [1.7, -0.3, 0],
        )

    return mesh


def coupled_rooms_office_to_stairwell(with_door=True):
    mesh = Mesh()
    office_z = 3.6
    stairwell_z = 14.4
    offset_x = -9.16
    offset_y = -2.95
    mesh.add_shoebox_with_coords([offset_x + 0, offset_y + 0, 0], 11.7, 5, office_z)
    mesh.add_shoebox_with_coords(
        [offset_x + 11.7, offset_y - 1.95, 0], 3.3, 6.3, stairwell_z
    )
    mesh.add_shoebox_with_coords([offset_x + 10.7, offset_y + 0, 0], 1, 0.8, office_z)
    mesh.add_shoebox_with_coords([offset_x + 10.7, offset_y + 3.5, 0], 1, 1.5, office_z)

    mesh.remove_patch_with_vertex_ids((20, 21, 5, 18))
    mesh.remove_patch_with_vertex_ids((17, 19, 1, 16))
    mesh.remove_patch_with_vertex_ids((27, 7, 26, 25))
    mesh.remove_patch_with_vertex_ids((24, 4, 23, 22))
    mesh.remove_patch_with_vertex_ids((18, 16, 1, 5))
    mesh.remove_patch_with_vertex_ids((7, 27, 24, 4))

    mesh.remove_patch_with_vertex_ids((14, 11, 8, 10))
    mesh.remove_patch_with_vertex_ids((7, 5, 1, 4))

    object_id = mesh.next_id
    mesh.add_parallelogram(
        [offset_x + 11.7, offset_y + 0, 0],
        [0, 0, office_z],
        [0, -1.95, 0],
        object_id=object_id,
    )
    mesh.add_parallelogram(
        [offset_x + 11.7, offset_y - 1.95, office_z],
        [0, 0, stairwell_z - office_z],
        [0, 6.3, 0],
        object_id=object_id,
    )
    if with_door:
        mesh.add_parallelogram(
            [offset_x + 11.7, offset_y + 0.8, 0],
            [0, 0, office_z],
            [0, 1.6, 0],
        )

    return mesh


def coupled_rooms_meeting_room_to_hallway(with_door=True):
    mesh = Mesh()
    z = 2.8
    offset_x = -2.1
    offset_y = -2.83
    mesh.add_shoebox_with_coords([offset_x + 0, offset_y + 0, 0], 4.6, 6.6, z)
    mesh.add_shoebox_with_coords([offset_x + 4.6, offset_y - 6.49, 0], 4.5, 6.49, z)
    mesh.add_shoebox_with_coords([offset_x + 4.6, offset_y, 0], 4.5, 6.6, z)
    mesh.add_shoebox_with_coords([offset_x + 4.6, offset_y + 6.6, 0], 4.5, 4.91, z)

    mesh.add_parallelogram([offset_x + 4.6, offset_y, 0], [0, 2.3, 0], [0, 0, z])
    mesh.add_parallelogram([offset_x + 4.6, offset_y + 6.6, 0], [0, -3.3, 0], [0, 0, z])

    if with_door:
        mesh.add_parallelogram(
            [offset_x + 4.6, offset_y + 6.6 - 3.3, 0], [1.0, 0, 0], [0, 0, z]
        )
    return mesh



def diffrir_classroom_base():
    inches = 0.0254
    max_x = 7.1247
    max_y = 7.9248
    max_z = 2.7432

    mesh = Mesh()
    mesh.add_shoebox_with_coords([0, 0, 0], max_x, max_y, max_z)
    mesh.add_parallelogram_with_coords(
        [0, 96 * inches, 29 * inches],
        [30 * inches, 96 * inches, 28.25 * inches],
        [0, max_y, 29 * inches],
    )
    mesh.add_parallelogram_with_coords(
        [max_x, 23.75 * inches, 29 * inches],
        [max_x, max_y, 29 * inches],
        [max_x - 30 * inches, 23.75 * inches, 28.25 * inches],
    )
    mesh.add_parallelogram_with_coords(
        [4.474256, 89 * inches, 29 * inches],
        [4.474256, max_y, 29 * inches],
        [2.935758, 89 * inches, 28.25 * inches],
    )
    return mesh


def diffrir_hallway_base():
    max_x = (1.532 + 1.526) / 2
    max_y = (18.042 + 18.154) / 2
    max_z = (2.746 + 2.765 + 2.766 + 2.756 + 2.749) / 5

    mesh = Mesh()
    mesh.add_shoebox_with_coords([0, 0, 0], max_x, max_y, max_z)
    return mesh


def diffrir_dampened_base():
    cm = 0.01
    max_x = 485 * cm
    max_y = 519.5 * cm
    max_z = 273.1 * cm

    mesh = Mesh()
    mesh.add_shoebox_with_coords([0, 0, 0], max_x, max_y, max_z)
    return mesh


def diffrir_complex_base():
    all_surfaces = []

    # Apex
    x_apex = 3.266
    z_apex = 6.086
    z_panel = 3.05
    x_to_door = 5.387

    # x_0_wall
    y_max = 13.014
    x_0_wall_z = 2.265
    x_0_wall = [[0, 0, 0], [0, y_max, 0], [0, 0, x_0_wall_z]]
    all_surfaces.append(x_0_wall)

    # y_0_wall
    x_max = 8.374
    y_0_wall_z = 2.644
    y_0_wall = [[0, 0, 0], [0, 0, y_0_wall_z], [x_max, 0, 0]]
    all_surfaces.append(y_0_wall)

    # y_0_wall_panel
    y_0_wall_panel = [[0, 0, z_panel], [0, 0, z_apex], [x_max, 0, z_panel]]
    all_surfaces.append(y_0_wall_panel)

    # y_max_wall_x0
    y_max_wall_x0 = [[0, y_max, 0], [x_max, y_max, 0], [0, y_max, z_panel]]
    all_surfaces.append(y_max_wall_x0)

    # y_max_wall_panel
    y_max_wall_panel = [
        [0, y_max, z_panel],
        [x_max, y_max, z_panel],
        [0, y_max, z_apex],
    ]
    all_surfaces.append(y_max_wall_panel)

    # y_max_wall_x_max
    y_door = 12.026
    z_door = 2.945
    y_door_wall_x_max = [
        [x_to_door, y_door, 0],
        [x_max, y_door, 0],
        [x_to_door, y_door, z_door],
    ]
    all_surfaces.append(y_door_wall_x_max)

    # glass_wall
    glass_wall = [
        [x_to_door, y_door, 0],
        [x_to_door, y_door, z_door],
        [x_to_door, y_max, 0],
    ]
    all_surfaces.append(glass_wall)

    # xmax_wall
    x_max_wall = [[x_max, y_door, 0], [x_max, 0, 0], [x_max, y_door, z_panel]]
    all_surfaces.append(x_max_wall)

    # Overhang
    y_overhang = 0.795
    z_1overhang = 2.644
    z_2overhang = 3.807

    # overhang_facing_down
    overhang_facing_down = [
        [0, 0, z_1overhang],
        [0, y_overhang, z_1overhang],
        [x_max, 0, z_1overhang],
    ]
    all_surfaces.append(overhang_facing_down)

    # overhang_facing_forward
    overhang_facing_forward = [
        [0, y_overhang, z_1overhang],
        [0, y_overhang, z_2overhang],
        [x_max, y_overhang, z_1overhang],
    ]
    all_surfaces.append(overhang_facing_forward)

    # overhang_facing_up
    overhang_facing_up = [
        [0, 0, z_2overhang],
        [x_max, 0, z_2overhang],
        [0, y_overhang, z_2overhang],
    ]
    all_surfaces.append(overhang_facing_up)

    # max_x_panels
    z_panel2 = 4.389
    x_max_wall = [[x_max, 0, z_panel], [x_max, 0, z_panel2], [x_max, y_max, z_panel]]
    all_surfaces.append(x_max_wall)

    # slant
    z_slant2 = 2.942
    x_slant = 0.920
    slant = [[0, 0, x_0_wall_z], [0, y_max, x_0_wall_z], [x_slant, 0, z_slant2]]
    all_surfaces.append(slant)

    # above_slant
    z_above_slant = 3.309
    above_slant = [
        [x_slant, 0, z_slant2],
        [x_slant, y_max, z_slant2],
        [x_slant, 0, z_above_slant],
    ]
    all_surfaces.append(above_slant)

    # floor
    floor = [[0, 0, 0], [x_max, 0, 0], [0, y_max, 0]]
    all_surfaces.append(floor)

    # door_top
    door_top = [
        [x_to_door, y_max, z_door],
        [x_to_door, y_door, z_door],
        [x_max, y_max, z_door],
    ]
    all_surfaces.append(door_top)

    ceiling_x0 = [
        [x_slant, 0, z_above_slant],
        [x_slant, y_max, z_above_slant],
        [x_apex, 0, z_apex],
    ]
    all_surfaces.append(ceiling_x0)

    # ceiling_x_max
    ceiling_x_max = [
        [x_max, 0, z_panel2],
        [x_apex, 0, z_apex],
        [x_max, y_max, z_panel2],
    ]
    all_surfaces.append(ceiling_x_max)

    # table
    x_width_table = 0.761
    y_table = 10.681
    z_table = 0.736
    table = [
        [x_max, 0, z_table],
        [x_max, y_table, z_table],
        [x_max - x_width_table, 0, z_table],
    ]
    all_surfaces.append(table)

    mesh = Mesh()
    for surface in all_surfaces:
        mesh.add_parallelogram_with_coords(*surface)

    pillar_1 = []
    # pillars
    x_pillar1 = 4.002
    x_width1 = 0.646
    y_width1 = 0.649
    y_pillar1 = 4.488
    z_pillar = 3.055

    # pillar1
    pillar1_1 = [
        [x_pillar1, y_pillar1, 0],
        [x_pillar1 + x_width1, y_pillar1, 0],
        [x_pillar1, y_pillar1, z_pillar],
    ]
    pillar_1.append(pillar1_1)

    pillar1_2 = [
        [x_pillar1, y_pillar1 + y_width1, 0],
        [x_pillar1, y_pillar1 + y_width1, z_pillar],
        [x_pillar1 + x_width1, y_pillar1 + y_width1, 0],
    ]
    pillar_1.append(pillar1_2)

    pillar1_3 = [
        [x_pillar1, y_pillar1, 0],
        [x_pillar1, y_pillar1, z_pillar],
        [x_pillar1, y_pillar1 + y_width1, 0],
    ]
    pillar_1.append(pillar1_3)

    pillar1_4 = [
        [x_pillar1 + x_width1, y_pillar1, 0],
        [x_pillar1 + x_width1, y_pillar1 + y_width1, 0],
        [x_pillar1 + x_width1, y_pillar1, z_pillar],
    ]
    pillar_1.append(pillar1_4)

    object_id = max(mesh.object_ids.values()) + 1
    for surface in pillar_1:
        mesh.add_parallelogram_with_coords(*surface, object_id=object_id)

    pillar_2 = []

    x_pillar2 = 4.004
    x_width2 = 0.654
    y_pillar2_1 = y_pillar1 + y_width1
    y_pillar2_2 = 7.178
    y_width2 = 0.937

    # pillar2
    pillar2_1 = [
        [x_pillar2, y_pillar2_1, 0],
        [x_pillar2 + x_width2, y_pillar2_1, 0],
        [x_pillar2, y_pillar2_2, z_pillar],
    ]
    pillar_2.append(pillar2_1)

    pillar2_2 = [
        [x_pillar2, y_pillar2_1 + y_width2, 0],
        [x_pillar2, y_pillar2_2 + y_width2, z_pillar],
        [x_pillar2 + x_width2, y_pillar2_1 + y_width2, 0],
    ]
    pillar_2.append(pillar2_2)

    pillar2_3 = [
        [x_pillar2, y_pillar2_1, 0],
        [x_pillar2, y_pillar2_2, z_pillar],
        [x_pillar2, y_pillar2_1 + y_width2, 0],
    ]
    pillar_2.append(pillar2_3)

    pillar2_4 = [
        [x_pillar2 + x_width2, y_pillar2_1, 0],
        [x_pillar2 + x_width2, y_pillar2_1 + y_width2, 0],
        [x_pillar2 + x_width2, y_pillar2_2, z_pillar],
    ]
    pillar_2.append(pillar2_4)

    object_id = max(mesh.object_ids.values()) + 1
    for surface in pillar_2:
        mesh.add_parallelogram_with_coords(*surface, object_id=object_id)

    pillar_3 = []

    # pillar3
    x_pillar3 = 3.997
    x_width3 = 0.647
    y_pillar3 = 10.249
    y_width3 = 0.649

    pillar3_1 = [
        [x_pillar3, y_pillar3, 0],
        [x_pillar3 + x_width3, y_pillar3, 0],
        [x_pillar3, y_pillar3, z_pillar],
    ]
    pillar_3.append(pillar3_1)

    pillar3_2 = [
        [x_pillar3, y_pillar3 + y_width3, 0],
        [x_pillar3, y_pillar3 + y_width3, z_pillar],
        [x_pillar3 + x_width3, y_pillar3 + y_width3, 0],
    ]
    pillar_3.append(pillar3_2)

    pillar3_3 = [
        [x_pillar3, y_pillar3, 0],
        [x_pillar3, y_pillar3, z_pillar],
        [x_pillar3, y_pillar3 + y_width3, 0],
    ]
    pillar_3.append(pillar3_3)

    pillar3_4 = [
        [x_pillar3 + x_width3, y_pillar3, 0],
        [x_pillar3 + x_width3, y_pillar3 + y_width3, 0],
        [x_pillar3 + x_width3, y_pillar3, z_pillar],
    ]
    pillar_3.append(pillar3_4)

    object_id = max(mesh.object_ids.values()) + 1
    for surface in pillar_3:
        mesh.add_parallelogram_with_coords(*surface, object_id=object_id)

    middle_table_z = 0.906
    middle_table_1 = [
        [2.927, 2.955, middle_table_z],
        [5.613, 4.378, middle_table_z],
        [x_pillar1 + 0.5 * x_width2, 5.444, middle_table_z],
    ]
    middle_table_2 = [
        [5.614, 7.738, middle_table_z],
        [3.217, 7.738, middle_table_z],
        [x_pillar1 + 0.5 * x_width2, 5.444, middle_table_z],
    ]
    for surface in [middle_table_1, middle_table_2]:
        mesh.add_triangle_with_coords(*surface)

    return mesh