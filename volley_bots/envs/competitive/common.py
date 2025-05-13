import torch
from carb import Float3
from typing import List,Tuple

def _carb_float3_add(a: Float3, b: Float3) -> Float3:
    return Float3(a.x + b.x, a.y + b.y, a.z + b.z)


def rectangular_cuboid_edges(length:float,width:float,z_low:float,height:float)->Tuple[List[Float3],List[Float3]]:
    """the rectangular cuboid is 
    """
    z=Float3(0,0,height)
    vertices=[
        Float3(-length/2,width/2,z_low),
        Float3(length/2,width/2,z_low),
        Float3(length/2,-width/2,z_low),
        Float3(-length/2,-width/2,z_low),
    ]
    points_start=[
        vertices[0],
            vertices[1],
            vertices[2],
            vertices[3],
            vertices[0],
            vertices[1],
            vertices[2],
            vertices[3],
            _carb_float3_add(vertices[0], z),
            _carb_float3_add(vertices[1] , z),
            _carb_float3_add(vertices[2] , z),
            _carb_float3_add(vertices[3] , z),
    ]
    
    points_end=[
        vertices[1],
            vertices[2],
            vertices[3],
            vertices[0],
            _carb_float3_add(vertices[0] , z),
            _carb_float3_add(vertices[1] , z),
            _carb_float3_add(vertices[2] , z),
            _carb_float3_add(vertices[3] , z),
            _carb_float3_add(vertices[1] , z),
            _carb_float3_add(vertices[2] , z),
            _carb_float3_add(vertices[3] , z),
            _carb_float3_add(vertices[0] , z),
    ]
    
    return points_start,points_end

