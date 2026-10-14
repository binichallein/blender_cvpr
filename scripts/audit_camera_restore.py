"""Run inside Blender with a loaded scene; compare camera transforms before/after restoration."""
import bpy,json
def info():
 return {o.name:{'matrix':list(map(list,o.matrix_world)), 'lens':o.data.lens,'parent':o.parent.name if o.parent else None,'constraints':[(c.name,c.type) for c in o.constraints]} for o in bpy.data.objects if o.type=='CAMERA'}
before=info()
copies={o.name:(o.matrix_world.copy(),o.data.copy()) for o in bpy.data.objects if o.type=='CAMERA'}
bpy.context.view_layer.update()
updated=info()
for name,(mat,data) in copies.items():
 o=bpy.data.objects[name];o.matrix_world=mat;o.data=data
bpy.context.view_layer.update()
after=info()
print('CAMERA_AUDIT',json.dumps({'before':before,'updated':updated,'after':after}))
