ц
??
:
Add
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
?
ResizeNearestNeighbor
images"T
size
resized_images"T"
Ttype:
2		"
align_cornersbool( "
half_pixel_centersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?
9
VarIsInitializedOp
resource
is_initialized
?"serve*2.1.02unknown??
?
inputPlaceholder*A
_output_shapes/
-:+???????????????????????????*
dtype0*6
shape-:+???????????????????????????
?
0conv2d_4/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@conv2d_4/kernel*
_output_shapes
:*
dtype0*%
valueB"             
?
.conv2d_4/kernel/Initializer/random_uniform/minConst*"
_class
loc:@conv2d_4/kernel*
_output_shapes
: *
dtype0*
valueB
 *???
?
.conv2d_4/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@conv2d_4/kernel*
_output_shapes
: *
dtype0*
valueB
 *??>
?
8conv2d_4/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_4/kernel/Initializer/random_uniform/shape*
T0*"
_class
loc:@conv2d_4/kernel*&
_output_shapes
: *
dtype0*

seed *
seed2 
?
.conv2d_4/kernel/Initializer/random_uniform/subSub.conv2d_4/kernel/Initializer/random_uniform/max.conv2d_4/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_4/kernel*
_output_shapes
: 
?
.conv2d_4/kernel/Initializer/random_uniform/mulMul8conv2d_4/kernel/Initializer/random_uniform/RandomUniform.conv2d_4/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv2d_4/kernel*&
_output_shapes
: 
?
*conv2d_4/kernel/Initializer/random_uniformAdd.conv2d_4/kernel/Initializer/random_uniform/mul.conv2d_4/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_4/kernel*&
_output_shapes
: 
?
conv2d_4/kernelVarHandleOp*"
_class
loc:@conv2d_4/kernel*
_output_shapes
: *
	container *
dtype0*
shape: * 
shared_nameconv2d_4/kernel
o
0conv2d_4/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_4/kernel*
_output_shapes
: 
t
conv2d_4/kernel/AssignAssignVariableOpconv2d_4/kernel*conv2d_4/kernel/Initializer/random_uniform*
dtype0
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
: *
dtype0
?
conv2d_4/bias/Initializer/zerosConst* 
_class
loc:@conv2d_4/bias*
_output_shapes
: *
dtype0*
valueB *    
?
conv2d_4/biasVarHandleOp* 
_class
loc:@conv2d_4/bias*
_output_shapes
: *
	container *
dtype0*
shape: *
shared_nameconv2d_4/bias
k
.conv2d_4/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_4/bias*
_output_shapes
: 
e
conv2d_4/bias/AssignAssignVariableOpconv2d_4/biasconv2d_4/bias/Initializer/zeros*
dtype0
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes
: *
dtype0
g
conv2d_4/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
v
conv2d_4/Conv2D/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
: *
dtype0
?
conv2d_4/Conv2DConv2Dinputconv2d_4/Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? *
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
i
conv2d_4/BiasAdd/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes
: *
dtype0
?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2Dconv2d_4/BiasAdd/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? *
data_formatNHWC
s
conv2d_4/ReluReluconv2d_4/BiasAdd*
T0*A
_output_shapes/
-:+??????????????????????????? 
?
0conv2d_5/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@conv2d_5/kernel*
_output_shapes
:*
dtype0*%
valueB"              
?
.conv2d_5/kernel/Initializer/random_uniform/minConst*"
_class
loc:@conv2d_5/kernel*
_output_shapes
: *
dtype0*
valueB
 *?ѽ
?
.conv2d_5/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@conv2d_5/kernel*
_output_shapes
: *
dtype0*
valueB
 *??=
?
8conv2d_5/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_5/kernel/Initializer/random_uniform/shape*
T0*"
_class
loc:@conv2d_5/kernel*&
_output_shapes
:  *
dtype0*

seed *
seed2 
?
.conv2d_5/kernel/Initializer/random_uniform/subSub.conv2d_5/kernel/Initializer/random_uniform/max.conv2d_5/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_5/kernel*
_output_shapes
: 
?
.conv2d_5/kernel/Initializer/random_uniform/mulMul8conv2d_5/kernel/Initializer/random_uniform/RandomUniform.conv2d_5/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv2d_5/kernel*&
_output_shapes
:  
?
*conv2d_5/kernel/Initializer/random_uniformAdd.conv2d_5/kernel/Initializer/random_uniform/mul.conv2d_5/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_5/kernel*&
_output_shapes
:  
?
conv2d_5/kernelVarHandleOp*"
_class
loc:@conv2d_5/kernel*
_output_shapes
: *
	container *
dtype0*
shape:  * 
shared_nameconv2d_5/kernel
o
0conv2d_5/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_5/kernel*
_output_shapes
: 
t
conv2d_5/kernel/AssignAssignVariableOpconv2d_5/kernel*conv2d_5/kernel/Initializer/random_uniform*
dtype0
{
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*&
_output_shapes
:  *
dtype0
?
conv2d_5/bias/Initializer/zerosConst* 
_class
loc:@conv2d_5/bias*
_output_shapes
: *
dtype0*
valueB *    
?
conv2d_5/biasVarHandleOp* 
_class
loc:@conv2d_5/bias*
_output_shapes
: *
	container *
dtype0*
shape: *
shared_nameconv2d_5/bias
k
.conv2d_5/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_5/bias*
_output_shapes
: 
e
conv2d_5/bias/AssignAssignVariableOpconv2d_5/biasconv2d_5/bias/Initializer/zeros*
dtype0
k
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes
: *
dtype0
g
conv2d_5/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
v
conv2d_5/Conv2D/ReadVariableOpReadVariableOpconv2d_5/kernel*&
_output_shapes
:  *
dtype0
?
conv2d_5/Conv2DConv2Dconv2d_4/Reluconv2d_5/Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? *
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
i
conv2d_5/BiasAdd/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes
: *
dtype0
?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2Dconv2d_5/BiasAdd/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? *
data_formatNHWC
s
conv2d_5/ReluReluconv2d_5/BiasAdd*
T0*A
_output_shapes/
-:+??????????????????????????? 
?
max_pooling2d_2/MaxPoolMaxPoolconv2d_5/Relu*
T0*A
_output_shapes/
-:+??????????????????????????? *
data_formatNHWC*
ksize
*
paddingVALID*
strides

?
9down_level_0_no_0/kernel/Initializer/random_uniform/shapeConst*+
_class!
loc:@down_level_0_no_0/kernel*
_output_shapes
:*
dtype0*%
valueB"              
?
7down_level_0_no_0/kernel/Initializer/random_uniform/minConst*+
_class!
loc:@down_level_0_no_0/kernel*
_output_shapes
: *
dtype0*
valueB
 *?ѽ
?
7down_level_0_no_0/kernel/Initializer/random_uniform/maxConst*+
_class!
loc:@down_level_0_no_0/kernel*
_output_shapes
: *
dtype0*
valueB
 *??=
?
Adown_level_0_no_0/kernel/Initializer/random_uniform/RandomUniformRandomUniform9down_level_0_no_0/kernel/Initializer/random_uniform/shape*
T0*+
_class!
loc:@down_level_0_no_0/kernel*&
_output_shapes
:  *
dtype0*

seed *
seed2 
?
7down_level_0_no_0/kernel/Initializer/random_uniform/subSub7down_level_0_no_0/kernel/Initializer/random_uniform/max7down_level_0_no_0/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@down_level_0_no_0/kernel*
_output_shapes
: 
?
7down_level_0_no_0/kernel/Initializer/random_uniform/mulMulAdown_level_0_no_0/kernel/Initializer/random_uniform/RandomUniform7down_level_0_no_0/kernel/Initializer/random_uniform/sub*
T0*+
_class!
loc:@down_level_0_no_0/kernel*&
_output_shapes
:  
?
3down_level_0_no_0/kernel/Initializer/random_uniformAdd7down_level_0_no_0/kernel/Initializer/random_uniform/mul7down_level_0_no_0/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@down_level_0_no_0/kernel*&
_output_shapes
:  
?
down_level_0_no_0/kernelVarHandleOp*+
_class!
loc:@down_level_0_no_0/kernel*
_output_shapes
: *
	container *
dtype0*
shape:  *)
shared_namedown_level_0_no_0/kernel
?
9down_level_0_no_0/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdown_level_0_no_0/kernel*
_output_shapes
: 
?
down_level_0_no_0/kernel/AssignAssignVariableOpdown_level_0_no_0/kernel3down_level_0_no_0/kernel/Initializer/random_uniform*
dtype0
?
,down_level_0_no_0/kernel/Read/ReadVariableOpReadVariableOpdown_level_0_no_0/kernel*&
_output_shapes
:  *
dtype0
?
(down_level_0_no_0/bias/Initializer/zerosConst*)
_class
loc:@down_level_0_no_0/bias*
_output_shapes
: *
dtype0*
valueB *    
?
down_level_0_no_0/biasVarHandleOp*)
_class
loc:@down_level_0_no_0/bias*
_output_shapes
: *
	container *
dtype0*
shape: *'
shared_namedown_level_0_no_0/bias
}
7down_level_0_no_0/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdown_level_0_no_0/bias*
_output_shapes
: 
?
down_level_0_no_0/bias/AssignAssignVariableOpdown_level_0_no_0/bias(down_level_0_no_0/bias/Initializer/zeros*
dtype0
}
*down_level_0_no_0/bias/Read/ReadVariableOpReadVariableOpdown_level_0_no_0/bias*
_output_shapes
: *
dtype0
p
down_level_0_no_0/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
?
'down_level_0_no_0/Conv2D/ReadVariableOpReadVariableOpdown_level_0_no_0/kernel*&
_output_shapes
:  *
dtype0
?
down_level_0_no_0/Conv2DConv2Dmax_pooling2d_2/MaxPool'down_level_0_no_0/Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? *
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
{
(down_level_0_no_0/BiasAdd/ReadVariableOpReadVariableOpdown_level_0_no_0/bias*
_output_shapes
: *
dtype0
?
down_level_0_no_0/BiasAddBiasAdddown_level_0_no_0/Conv2D(down_level_0_no_0/BiasAdd/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? *
data_formatNHWC
?
down_level_0_no_0/ReluReludown_level_0_no_0/BiasAdd*
T0*A
_output_shapes/
-:+??????????????????????????? 
?
9down_level_0_no_1/kernel/Initializer/random_uniform/shapeConst*+
_class!
loc:@down_level_0_no_1/kernel*
_output_shapes
:*
dtype0*%
valueB"              
?
7down_level_0_no_1/kernel/Initializer/random_uniform/minConst*+
_class!
loc:@down_level_0_no_1/kernel*
_output_shapes
: *
dtype0*
valueB
 *?ѽ
?
7down_level_0_no_1/kernel/Initializer/random_uniform/maxConst*+
_class!
loc:@down_level_0_no_1/kernel*
_output_shapes
: *
dtype0*
valueB
 *??=
?
Adown_level_0_no_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform9down_level_0_no_1/kernel/Initializer/random_uniform/shape*
T0*+
_class!
loc:@down_level_0_no_1/kernel*&
_output_shapes
:  *
dtype0*

seed *
seed2 
?
7down_level_0_no_1/kernel/Initializer/random_uniform/subSub7down_level_0_no_1/kernel/Initializer/random_uniform/max7down_level_0_no_1/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@down_level_0_no_1/kernel*
_output_shapes
: 
?
7down_level_0_no_1/kernel/Initializer/random_uniform/mulMulAdown_level_0_no_1/kernel/Initializer/random_uniform/RandomUniform7down_level_0_no_1/kernel/Initializer/random_uniform/sub*
T0*+
_class!
loc:@down_level_0_no_1/kernel*&
_output_shapes
:  
?
3down_level_0_no_1/kernel/Initializer/random_uniformAdd7down_level_0_no_1/kernel/Initializer/random_uniform/mul7down_level_0_no_1/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@down_level_0_no_1/kernel*&
_output_shapes
:  
?
down_level_0_no_1/kernelVarHandleOp*+
_class!
loc:@down_level_0_no_1/kernel*
_output_shapes
: *
	container *
dtype0*
shape:  *)
shared_namedown_level_0_no_1/kernel
?
9down_level_0_no_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdown_level_0_no_1/kernel*
_output_shapes
: 
?
down_level_0_no_1/kernel/AssignAssignVariableOpdown_level_0_no_1/kernel3down_level_0_no_1/kernel/Initializer/random_uniform*
dtype0
?
,down_level_0_no_1/kernel/Read/ReadVariableOpReadVariableOpdown_level_0_no_1/kernel*&
_output_shapes
:  *
dtype0
?
(down_level_0_no_1/bias/Initializer/zerosConst*)
_class
loc:@down_level_0_no_1/bias*
_output_shapes
: *
dtype0*
valueB *    
?
down_level_0_no_1/biasVarHandleOp*)
_class
loc:@down_level_0_no_1/bias*
_output_shapes
: *
	container *
dtype0*
shape: *'
shared_namedown_level_0_no_1/bias
}
7down_level_0_no_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdown_level_0_no_1/bias*
_output_shapes
: 
?
down_level_0_no_1/bias/AssignAssignVariableOpdown_level_0_no_1/bias(down_level_0_no_1/bias/Initializer/zeros*
dtype0
}
*down_level_0_no_1/bias/Read/ReadVariableOpReadVariableOpdown_level_0_no_1/bias*
_output_shapes
: *
dtype0
p
down_level_0_no_1/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
?
'down_level_0_no_1/Conv2D/ReadVariableOpReadVariableOpdown_level_0_no_1/kernel*&
_output_shapes
:  *
dtype0
?
down_level_0_no_1/Conv2DConv2Ddown_level_0_no_0/Relu'down_level_0_no_1/Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? *
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
{
(down_level_0_no_1/BiasAdd/ReadVariableOpReadVariableOpdown_level_0_no_1/bias*
_output_shapes
: *
dtype0
?
down_level_0_no_1/BiasAddBiasAdddown_level_0_no_1/Conv2D(down_level_0_no_1/BiasAdd/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? *
data_formatNHWC
?
down_level_0_no_1/ReluReludown_level_0_no_1/BiasAdd*
T0*A
_output_shapes/
-:+??????????????????????????? 
?
max_0/MaxPoolMaxPooldown_level_0_no_1/Relu*
T0*A
_output_shapes/
-:+??????????????????????????? *
data_formatNHWC*
ksize
*
paddingVALID*
strides

?
9down_level_1_no_0/kernel/Initializer/random_uniform/shapeConst*+
_class!
loc:@down_level_1_no_0/kernel*
_output_shapes
:*
dtype0*%
valueB"          @   
?
7down_level_1_no_0/kernel/Initializer/random_uniform/minConst*+
_class!
loc:@down_level_1_no_0/kernel*
_output_shapes
: *
dtype0*
valueB
 *????
?
7down_level_1_no_0/kernel/Initializer/random_uniform/maxConst*+
_class!
loc:@down_level_1_no_0/kernel*
_output_shapes
: *
dtype0*
valueB
 *???=
?
Adown_level_1_no_0/kernel/Initializer/random_uniform/RandomUniformRandomUniform9down_level_1_no_0/kernel/Initializer/random_uniform/shape*
T0*+
_class!
loc:@down_level_1_no_0/kernel*&
_output_shapes
: @*
dtype0*

seed *
seed2 
?
7down_level_1_no_0/kernel/Initializer/random_uniform/subSub7down_level_1_no_0/kernel/Initializer/random_uniform/max7down_level_1_no_0/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@down_level_1_no_0/kernel*
_output_shapes
: 
?
7down_level_1_no_0/kernel/Initializer/random_uniform/mulMulAdown_level_1_no_0/kernel/Initializer/random_uniform/RandomUniform7down_level_1_no_0/kernel/Initializer/random_uniform/sub*
T0*+
_class!
loc:@down_level_1_no_0/kernel*&
_output_shapes
: @
?
3down_level_1_no_0/kernel/Initializer/random_uniformAdd7down_level_1_no_0/kernel/Initializer/random_uniform/mul7down_level_1_no_0/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@down_level_1_no_0/kernel*&
_output_shapes
: @
?
down_level_1_no_0/kernelVarHandleOp*+
_class!
loc:@down_level_1_no_0/kernel*
_output_shapes
: *
	container *
dtype0*
shape: @*)
shared_namedown_level_1_no_0/kernel
?
9down_level_1_no_0/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdown_level_1_no_0/kernel*
_output_shapes
: 
?
down_level_1_no_0/kernel/AssignAssignVariableOpdown_level_1_no_0/kernel3down_level_1_no_0/kernel/Initializer/random_uniform*
dtype0
?
,down_level_1_no_0/kernel/Read/ReadVariableOpReadVariableOpdown_level_1_no_0/kernel*&
_output_shapes
: @*
dtype0
?
(down_level_1_no_0/bias/Initializer/zerosConst*)
_class
loc:@down_level_1_no_0/bias*
_output_shapes
:@*
dtype0*
valueB@*    
?
down_level_1_no_0/biasVarHandleOp*)
_class
loc:@down_level_1_no_0/bias*
_output_shapes
: *
	container *
dtype0*
shape:@*'
shared_namedown_level_1_no_0/bias
}
7down_level_1_no_0/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdown_level_1_no_0/bias*
_output_shapes
: 
?
down_level_1_no_0/bias/AssignAssignVariableOpdown_level_1_no_0/bias(down_level_1_no_0/bias/Initializer/zeros*
dtype0
}
*down_level_1_no_0/bias/Read/ReadVariableOpReadVariableOpdown_level_1_no_0/bias*
_output_shapes
:@*
dtype0
p
down_level_1_no_0/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
?
'down_level_1_no_0/Conv2D/ReadVariableOpReadVariableOpdown_level_1_no_0/kernel*&
_output_shapes
: @*
dtype0
?
down_level_1_no_0/Conv2DConv2Dmax_0/MaxPool'down_level_1_no_0/Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
{
(down_level_1_no_0/BiasAdd/ReadVariableOpReadVariableOpdown_level_1_no_0/bias*
_output_shapes
:@*
dtype0
?
down_level_1_no_0/BiasAddBiasAdddown_level_1_no_0/Conv2D(down_level_1_no_0/BiasAdd/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@*
data_formatNHWC
?
down_level_1_no_0/ReluReludown_level_1_no_0/BiasAdd*
T0*A
_output_shapes/
-:+???????????????????????????@
?
9down_level_1_no_1/kernel/Initializer/random_uniform/shapeConst*+
_class!
loc:@down_level_1_no_1/kernel*
_output_shapes
:*
dtype0*%
valueB"      @   @   
?
7down_level_1_no_1/kernel/Initializer/random_uniform/minConst*+
_class!
loc:@down_level_1_no_1/kernel*
_output_shapes
: *
dtype0*
valueB
 *:͓?
?
7down_level_1_no_1/kernel/Initializer/random_uniform/maxConst*+
_class!
loc:@down_level_1_no_1/kernel*
_output_shapes
: *
dtype0*
valueB
 *:͓=
?
Adown_level_1_no_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform9down_level_1_no_1/kernel/Initializer/random_uniform/shape*
T0*+
_class!
loc:@down_level_1_no_1/kernel*&
_output_shapes
:@@*
dtype0*

seed *
seed2 
?
7down_level_1_no_1/kernel/Initializer/random_uniform/subSub7down_level_1_no_1/kernel/Initializer/random_uniform/max7down_level_1_no_1/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@down_level_1_no_1/kernel*
_output_shapes
: 
?
7down_level_1_no_1/kernel/Initializer/random_uniform/mulMulAdown_level_1_no_1/kernel/Initializer/random_uniform/RandomUniform7down_level_1_no_1/kernel/Initializer/random_uniform/sub*
T0*+
_class!
loc:@down_level_1_no_1/kernel*&
_output_shapes
:@@
?
3down_level_1_no_1/kernel/Initializer/random_uniformAdd7down_level_1_no_1/kernel/Initializer/random_uniform/mul7down_level_1_no_1/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@down_level_1_no_1/kernel*&
_output_shapes
:@@
?
down_level_1_no_1/kernelVarHandleOp*+
_class!
loc:@down_level_1_no_1/kernel*
_output_shapes
: *
	container *
dtype0*
shape:@@*)
shared_namedown_level_1_no_1/kernel
?
9down_level_1_no_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdown_level_1_no_1/kernel*
_output_shapes
: 
?
down_level_1_no_1/kernel/AssignAssignVariableOpdown_level_1_no_1/kernel3down_level_1_no_1/kernel/Initializer/random_uniform*
dtype0
?
,down_level_1_no_1/kernel/Read/ReadVariableOpReadVariableOpdown_level_1_no_1/kernel*&
_output_shapes
:@@*
dtype0
?
(down_level_1_no_1/bias/Initializer/zerosConst*)
_class
loc:@down_level_1_no_1/bias*
_output_shapes
:@*
dtype0*
valueB@*    
?
down_level_1_no_1/biasVarHandleOp*)
_class
loc:@down_level_1_no_1/bias*
_output_shapes
: *
	container *
dtype0*
shape:@*'
shared_namedown_level_1_no_1/bias
}
7down_level_1_no_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdown_level_1_no_1/bias*
_output_shapes
: 
?
down_level_1_no_1/bias/AssignAssignVariableOpdown_level_1_no_1/bias(down_level_1_no_1/bias/Initializer/zeros*
dtype0
}
*down_level_1_no_1/bias/Read/ReadVariableOpReadVariableOpdown_level_1_no_1/bias*
_output_shapes
:@*
dtype0
p
down_level_1_no_1/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
?
'down_level_1_no_1/Conv2D/ReadVariableOpReadVariableOpdown_level_1_no_1/kernel*&
_output_shapes
:@@*
dtype0
?
down_level_1_no_1/Conv2DConv2Ddown_level_1_no_0/Relu'down_level_1_no_1/Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
{
(down_level_1_no_1/BiasAdd/ReadVariableOpReadVariableOpdown_level_1_no_1/bias*
_output_shapes
:@*
dtype0
?
down_level_1_no_1/BiasAddBiasAdddown_level_1_no_1/Conv2D(down_level_1_no_1/BiasAdd/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@*
data_formatNHWC
?
down_level_1_no_1/ReluReludown_level_1_no_1/BiasAdd*
T0*A
_output_shapes/
-:+???????????????????????????@
?
max_1/MaxPoolMaxPooldown_level_1_no_1/Relu*
T0*A
_output_shapes/
-:+???????????????????????????@*
data_formatNHWC*
ksize
*
paddingVALID*
strides

?
9down_level_2_no_0/kernel/Initializer/random_uniform/shapeConst*+
_class!
loc:@down_level_2_no_0/kernel*
_output_shapes
:*
dtype0*%
valueB"      @   ?   
?
7down_level_2_no_0/kernel/Initializer/random_uniform/minConst*+
_class!
loc:@down_level_2_no_0/kernel*
_output_shapes
: *
dtype0*
valueB
 *?[q?
?
7down_level_2_no_0/kernel/Initializer/random_uniform/maxConst*+
_class!
loc:@down_level_2_no_0/kernel*
_output_shapes
: *
dtype0*
valueB
 *?[q=
?
Adown_level_2_no_0/kernel/Initializer/random_uniform/RandomUniformRandomUniform9down_level_2_no_0/kernel/Initializer/random_uniform/shape*
T0*+
_class!
loc:@down_level_2_no_0/kernel*'
_output_shapes
:@?*
dtype0*

seed *
seed2 
?
7down_level_2_no_0/kernel/Initializer/random_uniform/subSub7down_level_2_no_0/kernel/Initializer/random_uniform/max7down_level_2_no_0/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@down_level_2_no_0/kernel*
_output_shapes
: 
?
7down_level_2_no_0/kernel/Initializer/random_uniform/mulMulAdown_level_2_no_0/kernel/Initializer/random_uniform/RandomUniform7down_level_2_no_0/kernel/Initializer/random_uniform/sub*
T0*+
_class!
loc:@down_level_2_no_0/kernel*'
_output_shapes
:@?
?
3down_level_2_no_0/kernel/Initializer/random_uniformAdd7down_level_2_no_0/kernel/Initializer/random_uniform/mul7down_level_2_no_0/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@down_level_2_no_0/kernel*'
_output_shapes
:@?
?
down_level_2_no_0/kernelVarHandleOp*+
_class!
loc:@down_level_2_no_0/kernel*
_output_shapes
: *
	container *
dtype0*
shape:@?*)
shared_namedown_level_2_no_0/kernel
?
9down_level_2_no_0/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdown_level_2_no_0/kernel*
_output_shapes
: 
?
down_level_2_no_0/kernel/AssignAssignVariableOpdown_level_2_no_0/kernel3down_level_2_no_0/kernel/Initializer/random_uniform*
dtype0
?
,down_level_2_no_0/kernel/Read/ReadVariableOpReadVariableOpdown_level_2_no_0/kernel*'
_output_shapes
:@?*
dtype0
?
(down_level_2_no_0/bias/Initializer/zerosConst*)
_class
loc:@down_level_2_no_0/bias*
_output_shapes	
:?*
dtype0*
valueB?*    
?
down_level_2_no_0/biasVarHandleOp*)
_class
loc:@down_level_2_no_0/bias*
_output_shapes
: *
	container *
dtype0*
shape:?*'
shared_namedown_level_2_no_0/bias
}
7down_level_2_no_0/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdown_level_2_no_0/bias*
_output_shapes
: 
?
down_level_2_no_0/bias/AssignAssignVariableOpdown_level_2_no_0/bias(down_level_2_no_0/bias/Initializer/zeros*
dtype0
~
*down_level_2_no_0/bias/Read/ReadVariableOpReadVariableOpdown_level_2_no_0/bias*
_output_shapes	
:?*
dtype0
p
down_level_2_no_0/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
?
'down_level_2_no_0/Conv2D/ReadVariableOpReadVariableOpdown_level_2_no_0/kernel*'
_output_shapes
:@?*
dtype0
?
down_level_2_no_0/Conv2DConv2Dmax_1/MaxPool'down_level_2_no_0/Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
|
(down_level_2_no_0/BiasAdd/ReadVariableOpReadVariableOpdown_level_2_no_0/bias*
_output_shapes	
:?*
dtype0
?
down_level_2_no_0/BiasAddBiasAdddown_level_2_no_0/Conv2D(down_level_2_no_0/BiasAdd/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????*
data_formatNHWC
?
down_level_2_no_0/ReluReludown_level_2_no_0/BiasAdd*
T0*B
_output_shapes0
.:,????????????????????????????
?
9down_level_2_no_1/kernel/Initializer/random_uniform/shapeConst*+
_class!
loc:@down_level_2_no_1/kernel*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   
?
7down_level_2_no_1/kernel/Initializer/random_uniform/minConst*+
_class!
loc:@down_level_2_no_1/kernel*
_output_shapes
: *
dtype0*
valueB
 *?Q?
?
7down_level_2_no_1/kernel/Initializer/random_uniform/maxConst*+
_class!
loc:@down_level_2_no_1/kernel*
_output_shapes
: *
dtype0*
valueB
 *?Q=
?
Adown_level_2_no_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform9down_level_2_no_1/kernel/Initializer/random_uniform/shape*
T0*+
_class!
loc:@down_level_2_no_1/kernel*(
_output_shapes
:??*
dtype0*

seed *
seed2 
?
7down_level_2_no_1/kernel/Initializer/random_uniform/subSub7down_level_2_no_1/kernel/Initializer/random_uniform/max7down_level_2_no_1/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@down_level_2_no_1/kernel*
_output_shapes
: 
?
7down_level_2_no_1/kernel/Initializer/random_uniform/mulMulAdown_level_2_no_1/kernel/Initializer/random_uniform/RandomUniform7down_level_2_no_1/kernel/Initializer/random_uniform/sub*
T0*+
_class!
loc:@down_level_2_no_1/kernel*(
_output_shapes
:??
?
3down_level_2_no_1/kernel/Initializer/random_uniformAdd7down_level_2_no_1/kernel/Initializer/random_uniform/mul7down_level_2_no_1/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@down_level_2_no_1/kernel*(
_output_shapes
:??
?
down_level_2_no_1/kernelVarHandleOp*+
_class!
loc:@down_level_2_no_1/kernel*
_output_shapes
: *
	container *
dtype0*
shape:??*)
shared_namedown_level_2_no_1/kernel
?
9down_level_2_no_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdown_level_2_no_1/kernel*
_output_shapes
: 
?
down_level_2_no_1/kernel/AssignAssignVariableOpdown_level_2_no_1/kernel3down_level_2_no_1/kernel/Initializer/random_uniform*
dtype0
?
,down_level_2_no_1/kernel/Read/ReadVariableOpReadVariableOpdown_level_2_no_1/kernel*(
_output_shapes
:??*
dtype0
?
(down_level_2_no_1/bias/Initializer/zerosConst*)
_class
loc:@down_level_2_no_1/bias*
_output_shapes	
:?*
dtype0*
valueB?*    
?
down_level_2_no_1/biasVarHandleOp*)
_class
loc:@down_level_2_no_1/bias*
_output_shapes
: *
	container *
dtype0*
shape:?*'
shared_namedown_level_2_no_1/bias
}
7down_level_2_no_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdown_level_2_no_1/bias*
_output_shapes
: 
?
down_level_2_no_1/bias/AssignAssignVariableOpdown_level_2_no_1/bias(down_level_2_no_1/bias/Initializer/zeros*
dtype0
~
*down_level_2_no_1/bias/Read/ReadVariableOpReadVariableOpdown_level_2_no_1/bias*
_output_shapes	
:?*
dtype0
p
down_level_2_no_1/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
?
'down_level_2_no_1/Conv2D/ReadVariableOpReadVariableOpdown_level_2_no_1/kernel*(
_output_shapes
:??*
dtype0
?
down_level_2_no_1/Conv2DConv2Ddown_level_2_no_0/Relu'down_level_2_no_1/Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
|
(down_level_2_no_1/BiasAdd/ReadVariableOpReadVariableOpdown_level_2_no_1/bias*
_output_shapes	
:?*
dtype0
?
down_level_2_no_1/BiasAddBiasAdddown_level_2_no_1/Conv2D(down_level_2_no_1/BiasAdd/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????*
data_formatNHWC
?
down_level_2_no_1/ReluReludown_level_2_no_1/BiasAdd*
T0*B
_output_shapes0
.:,????????????????????????????
?
max_2/MaxPoolMaxPooldown_level_2_no_1/Relu*
T0*B
_output_shapes0
.:,????????????????????????????*
data_formatNHWC*
ksize
*
paddingVALID*
strides

?
0middle_0/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@middle_0/kernel*
_output_shapes
:*
dtype0*%
valueB"      ?      
?
.middle_0/kernel/Initializer/random_uniform/minConst*"
_class
loc:@middle_0/kernel*
_output_shapes
: *
dtype0*
valueB
 *??*?
?
.middle_0/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@middle_0/kernel*
_output_shapes
: *
dtype0*
valueB
 *??*=
?
8middle_0/kernel/Initializer/random_uniform/RandomUniformRandomUniform0middle_0/kernel/Initializer/random_uniform/shape*
T0*"
_class
loc:@middle_0/kernel*(
_output_shapes
:??*
dtype0*

seed *
seed2 
?
.middle_0/kernel/Initializer/random_uniform/subSub.middle_0/kernel/Initializer/random_uniform/max.middle_0/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@middle_0/kernel*
_output_shapes
: 
?
.middle_0/kernel/Initializer/random_uniform/mulMul8middle_0/kernel/Initializer/random_uniform/RandomUniform.middle_0/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@middle_0/kernel*(
_output_shapes
:??
?
*middle_0/kernel/Initializer/random_uniformAdd.middle_0/kernel/Initializer/random_uniform/mul.middle_0/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@middle_0/kernel*(
_output_shapes
:??
?
middle_0/kernelVarHandleOp*"
_class
loc:@middle_0/kernel*
_output_shapes
: *
	container *
dtype0*
shape:??* 
shared_namemiddle_0/kernel
o
0middle_0/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpmiddle_0/kernel*
_output_shapes
: 
t
middle_0/kernel/AssignAssignVariableOpmiddle_0/kernel*middle_0/kernel/Initializer/random_uniform*
dtype0
}
#middle_0/kernel/Read/ReadVariableOpReadVariableOpmiddle_0/kernel*(
_output_shapes
:??*
dtype0
?
middle_0/bias/Initializer/zerosConst* 
_class
loc:@middle_0/bias*
_output_shapes	
:?*
dtype0*
valueB?*    
?
middle_0/biasVarHandleOp* 
_class
loc:@middle_0/bias*
_output_shapes
: *
	container *
dtype0*
shape:?*
shared_namemiddle_0/bias
k
.middle_0/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpmiddle_0/bias*
_output_shapes
: 
e
middle_0/bias/AssignAssignVariableOpmiddle_0/biasmiddle_0/bias/Initializer/zeros*
dtype0
l
!middle_0/bias/Read/ReadVariableOpReadVariableOpmiddle_0/bias*
_output_shapes	
:?*
dtype0
g
middle_0/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
x
middle_0/Conv2D/ReadVariableOpReadVariableOpmiddle_0/kernel*(
_output_shapes
:??*
dtype0
?
middle_0/Conv2DConv2Dmax_2/MaxPoolmiddle_0/Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
j
middle_0/BiasAdd/ReadVariableOpReadVariableOpmiddle_0/bias*
_output_shapes	
:?*
dtype0
?
middle_0/BiasAddBiasAddmiddle_0/Conv2Dmiddle_0/BiasAdd/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????*
data_formatNHWC
t
middle_0/ReluRelumiddle_0/BiasAdd*
T0*B
_output_shapes0
.:,????????????????????????????
?
0middle_2/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@middle_2/kernel*
_output_shapes
:*
dtype0*%
valueB"         ?   
?
.middle_2/kernel/Initializer/random_uniform/minConst*"
_class
loc:@middle_2/kernel*
_output_shapes
: *
dtype0*
valueB
 *??*?
?
.middle_2/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@middle_2/kernel*
_output_shapes
: *
dtype0*
valueB
 *??*=
?
8middle_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform0middle_2/kernel/Initializer/random_uniform/shape*
T0*"
_class
loc:@middle_2/kernel*(
_output_shapes
:??*
dtype0*

seed *
seed2 
?
.middle_2/kernel/Initializer/random_uniform/subSub.middle_2/kernel/Initializer/random_uniform/max.middle_2/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@middle_2/kernel*
_output_shapes
: 
?
.middle_2/kernel/Initializer/random_uniform/mulMul8middle_2/kernel/Initializer/random_uniform/RandomUniform.middle_2/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@middle_2/kernel*(
_output_shapes
:??
?
*middle_2/kernel/Initializer/random_uniformAdd.middle_2/kernel/Initializer/random_uniform/mul.middle_2/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@middle_2/kernel*(
_output_shapes
:??
?
middle_2/kernelVarHandleOp*"
_class
loc:@middle_2/kernel*
_output_shapes
: *
	container *
dtype0*
shape:??* 
shared_namemiddle_2/kernel
o
0middle_2/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpmiddle_2/kernel*
_output_shapes
: 
t
middle_2/kernel/AssignAssignVariableOpmiddle_2/kernel*middle_2/kernel/Initializer/random_uniform*
dtype0
}
#middle_2/kernel/Read/ReadVariableOpReadVariableOpmiddle_2/kernel*(
_output_shapes
:??*
dtype0
?
middle_2/bias/Initializer/zerosConst* 
_class
loc:@middle_2/bias*
_output_shapes	
:?*
dtype0*
valueB?*    
?
middle_2/biasVarHandleOp* 
_class
loc:@middle_2/bias*
_output_shapes
: *
	container *
dtype0*
shape:?*
shared_namemiddle_2/bias
k
.middle_2/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpmiddle_2/bias*
_output_shapes
: 
e
middle_2/bias/AssignAssignVariableOpmiddle_2/biasmiddle_2/bias/Initializer/zeros*
dtype0
l
!middle_2/bias/Read/ReadVariableOpReadVariableOpmiddle_2/bias*
_output_shapes	
:?*
dtype0
g
middle_2/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
x
middle_2/Conv2D/ReadVariableOpReadVariableOpmiddle_2/kernel*(
_output_shapes
:??*
dtype0
?
middle_2/Conv2DConv2Dmiddle_0/Relumiddle_2/Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
j
middle_2/BiasAdd/ReadVariableOpReadVariableOpmiddle_2/bias*
_output_shapes	
:?*
dtype0
?
middle_2/BiasAddBiasAddmiddle_2/Conv2Dmiddle_2/BiasAdd/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????*
data_formatNHWC
t
middle_2/ReluRelumiddle_2/BiasAdd*
T0*B
_output_shapes0
.:,????????????????????????????
b
up_sampling2d_6/ShapeShapemiddle_2/Relu*
T0*
_output_shapes
:*
out_type0
m
#up_sampling2d_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
o
%up_sampling2d_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
o
%up_sampling2d_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
up_sampling2d_6/strided_sliceStridedSliceup_sampling2d_6/Shape#up_sampling2d_6/strided_slice/stack%up_sampling2d_6/strided_slice/stack_1%up_sampling2d_6/strided_slice/stack_2*
Index0*
T0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask 
f
up_sampling2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"      
u
up_sampling2d_6/mulMulup_sampling2d_6/strided_sliceup_sampling2d_6/Const*
T0*
_output_shapes
:
?
,up_sampling2d_6/resize/ResizeNearestNeighborResizeNearestNeighbormiddle_2/Reluup_sampling2d_6/mul*
T0*B
_output_shapes0
.:,????????????????????????????*
align_corners( *
half_pixel_centers(
[
concatenate_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
?
concatenate_6/concatConcatV2,up_sampling2d_6/resize/ResizeNearestNeighbordown_level_2_no_1/Reluconcatenate_6/concat/axis*
N*
T0*

Tidx0*B
_output_shapes0
.:,????????????????????????????
?
7up_level_2_no_0/kernel/Initializer/random_uniform/shapeConst*)
_class
loc:@up_level_2_no_0/kernel*
_output_shapes
:*
dtype0*%
valueB"         ?   
?
5up_level_2_no_0/kernel/Initializer/random_uniform/minConst*)
_class
loc:@up_level_2_no_0/kernel*
_output_shapes
: *
dtype0*
valueB
 *??*?
?
5up_level_2_no_0/kernel/Initializer/random_uniform/maxConst*)
_class
loc:@up_level_2_no_0/kernel*
_output_shapes
: *
dtype0*
valueB
 *??*=
?
?up_level_2_no_0/kernel/Initializer/random_uniform/RandomUniformRandomUniform7up_level_2_no_0/kernel/Initializer/random_uniform/shape*
T0*)
_class
loc:@up_level_2_no_0/kernel*(
_output_shapes
:??*
dtype0*

seed *
seed2 
?
5up_level_2_no_0/kernel/Initializer/random_uniform/subSub5up_level_2_no_0/kernel/Initializer/random_uniform/max5up_level_2_no_0/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@up_level_2_no_0/kernel*
_output_shapes
: 
?
5up_level_2_no_0/kernel/Initializer/random_uniform/mulMul?up_level_2_no_0/kernel/Initializer/random_uniform/RandomUniform5up_level_2_no_0/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@up_level_2_no_0/kernel*(
_output_shapes
:??
?
1up_level_2_no_0/kernel/Initializer/random_uniformAdd5up_level_2_no_0/kernel/Initializer/random_uniform/mul5up_level_2_no_0/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@up_level_2_no_0/kernel*(
_output_shapes
:??
?
up_level_2_no_0/kernelVarHandleOp*)
_class
loc:@up_level_2_no_0/kernel*
_output_shapes
: *
	container *
dtype0*
shape:??*'
shared_nameup_level_2_no_0/kernel
}
7up_level_2_no_0/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpup_level_2_no_0/kernel*
_output_shapes
: 
?
up_level_2_no_0/kernel/AssignAssignVariableOpup_level_2_no_0/kernel1up_level_2_no_0/kernel/Initializer/random_uniform*
dtype0
?
*up_level_2_no_0/kernel/Read/ReadVariableOpReadVariableOpup_level_2_no_0/kernel*(
_output_shapes
:??*
dtype0
?
&up_level_2_no_0/bias/Initializer/zerosConst*'
_class
loc:@up_level_2_no_0/bias*
_output_shapes	
:?*
dtype0*
valueB?*    
?
up_level_2_no_0/biasVarHandleOp*'
_class
loc:@up_level_2_no_0/bias*
_output_shapes
: *
	container *
dtype0*
shape:?*%
shared_nameup_level_2_no_0/bias
y
5up_level_2_no_0/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpup_level_2_no_0/bias*
_output_shapes
: 
z
up_level_2_no_0/bias/AssignAssignVariableOpup_level_2_no_0/bias&up_level_2_no_0/bias/Initializer/zeros*
dtype0
z
(up_level_2_no_0/bias/Read/ReadVariableOpReadVariableOpup_level_2_no_0/bias*
_output_shapes	
:?*
dtype0
n
up_level_2_no_0/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
?
%up_level_2_no_0/Conv2D/ReadVariableOpReadVariableOpup_level_2_no_0/kernel*(
_output_shapes
:??*
dtype0
?
up_level_2_no_0/Conv2DConv2Dconcatenate_6/concat%up_level_2_no_0/Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
x
&up_level_2_no_0/BiasAdd/ReadVariableOpReadVariableOpup_level_2_no_0/bias*
_output_shapes	
:?*
dtype0
?
up_level_2_no_0/BiasAddBiasAddup_level_2_no_0/Conv2D&up_level_2_no_0/BiasAdd/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????*
data_formatNHWC
?
up_level_2_no_0/ReluReluup_level_2_no_0/BiasAdd*
T0*B
_output_shapes0
.:,????????????????????????????
?
7up_level_2_no_2/kernel/Initializer/random_uniform/shapeConst*)
_class
loc:@up_level_2_no_2/kernel*
_output_shapes
:*
dtype0*%
valueB"      ?   @   
?
5up_level_2_no_2/kernel/Initializer/random_uniform/minConst*)
_class
loc:@up_level_2_no_2/kernel*
_output_shapes
: *
dtype0*
valueB
 *?[q?
?
5up_level_2_no_2/kernel/Initializer/random_uniform/maxConst*)
_class
loc:@up_level_2_no_2/kernel*
_output_shapes
: *
dtype0*
valueB
 *?[q=
?
?up_level_2_no_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform7up_level_2_no_2/kernel/Initializer/random_uniform/shape*
T0*)
_class
loc:@up_level_2_no_2/kernel*'
_output_shapes
:?@*
dtype0*

seed *
seed2 
?
5up_level_2_no_2/kernel/Initializer/random_uniform/subSub5up_level_2_no_2/kernel/Initializer/random_uniform/max5up_level_2_no_2/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@up_level_2_no_2/kernel*
_output_shapes
: 
?
5up_level_2_no_2/kernel/Initializer/random_uniform/mulMul?up_level_2_no_2/kernel/Initializer/random_uniform/RandomUniform5up_level_2_no_2/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@up_level_2_no_2/kernel*'
_output_shapes
:?@
?
1up_level_2_no_2/kernel/Initializer/random_uniformAdd5up_level_2_no_2/kernel/Initializer/random_uniform/mul5up_level_2_no_2/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@up_level_2_no_2/kernel*'
_output_shapes
:?@
?
up_level_2_no_2/kernelVarHandleOp*)
_class
loc:@up_level_2_no_2/kernel*
_output_shapes
: *
	container *
dtype0*
shape:?@*'
shared_nameup_level_2_no_2/kernel
}
7up_level_2_no_2/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpup_level_2_no_2/kernel*
_output_shapes
: 
?
up_level_2_no_2/kernel/AssignAssignVariableOpup_level_2_no_2/kernel1up_level_2_no_2/kernel/Initializer/random_uniform*
dtype0
?
*up_level_2_no_2/kernel/Read/ReadVariableOpReadVariableOpup_level_2_no_2/kernel*'
_output_shapes
:?@*
dtype0
?
&up_level_2_no_2/bias/Initializer/zerosConst*'
_class
loc:@up_level_2_no_2/bias*
_output_shapes
:@*
dtype0*
valueB@*    
?
up_level_2_no_2/biasVarHandleOp*'
_class
loc:@up_level_2_no_2/bias*
_output_shapes
: *
	container *
dtype0*
shape:@*%
shared_nameup_level_2_no_2/bias
y
5up_level_2_no_2/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpup_level_2_no_2/bias*
_output_shapes
: 
z
up_level_2_no_2/bias/AssignAssignVariableOpup_level_2_no_2/bias&up_level_2_no_2/bias/Initializer/zeros*
dtype0
y
(up_level_2_no_2/bias/Read/ReadVariableOpReadVariableOpup_level_2_no_2/bias*
_output_shapes
:@*
dtype0
n
up_level_2_no_2/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
?
%up_level_2_no_2/Conv2D/ReadVariableOpReadVariableOpup_level_2_no_2/kernel*'
_output_shapes
:?@*
dtype0
?
up_level_2_no_2/Conv2DConv2Dup_level_2_no_0/Relu%up_level_2_no_2/Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
w
&up_level_2_no_2/BiasAdd/ReadVariableOpReadVariableOpup_level_2_no_2/bias*
_output_shapes
:@*
dtype0
?
up_level_2_no_2/BiasAddBiasAddup_level_2_no_2/Conv2D&up_level_2_no_2/BiasAdd/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@*
data_formatNHWC
?
up_level_2_no_2/ReluReluup_level_2_no_2/BiasAdd*
T0*A
_output_shapes/
-:+???????????????????????????@
i
up_sampling2d_7/ShapeShapeup_level_2_no_2/Relu*
T0*
_output_shapes
:*
out_type0
m
#up_sampling2d_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
o
%up_sampling2d_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
o
%up_sampling2d_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
up_sampling2d_7/strided_sliceStridedSliceup_sampling2d_7/Shape#up_sampling2d_7/strided_slice/stack%up_sampling2d_7/strided_slice/stack_1%up_sampling2d_7/strided_slice/stack_2*
Index0*
T0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask 
f
up_sampling2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"      
u
up_sampling2d_7/mulMulup_sampling2d_7/strided_sliceup_sampling2d_7/Const*
T0*
_output_shapes
:
?
,up_sampling2d_7/resize/ResizeNearestNeighborResizeNearestNeighborup_level_2_no_2/Reluup_sampling2d_7/mul*
T0*A
_output_shapes/
-:+???????????????????????????@*
align_corners( *
half_pixel_centers(
[
concatenate_7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
?
concatenate_7/concatConcatV2,up_sampling2d_7/resize/ResizeNearestNeighbordown_level_1_no_1/Reluconcatenate_7/concat/axis*
N*
T0*

Tidx0*B
_output_shapes0
.:,????????????????????????????
?
7up_level_1_no_0/kernel/Initializer/random_uniform/shapeConst*)
_class
loc:@up_level_1_no_0/kernel*
_output_shapes
:*
dtype0*%
valueB"      ?   @   
?
5up_level_1_no_0/kernel/Initializer/random_uniform/minConst*)
_class
loc:@up_level_1_no_0/kernel*
_output_shapes
: *
dtype0*
valueB
 *?[q?
?
5up_level_1_no_0/kernel/Initializer/random_uniform/maxConst*)
_class
loc:@up_level_1_no_0/kernel*
_output_shapes
: *
dtype0*
valueB
 *?[q=
?
?up_level_1_no_0/kernel/Initializer/random_uniform/RandomUniformRandomUniform7up_level_1_no_0/kernel/Initializer/random_uniform/shape*
T0*)
_class
loc:@up_level_1_no_0/kernel*'
_output_shapes
:?@*
dtype0*

seed *
seed2 
?
5up_level_1_no_0/kernel/Initializer/random_uniform/subSub5up_level_1_no_0/kernel/Initializer/random_uniform/max5up_level_1_no_0/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@up_level_1_no_0/kernel*
_output_shapes
: 
?
5up_level_1_no_0/kernel/Initializer/random_uniform/mulMul?up_level_1_no_0/kernel/Initializer/random_uniform/RandomUniform5up_level_1_no_0/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@up_level_1_no_0/kernel*'
_output_shapes
:?@
?
1up_level_1_no_0/kernel/Initializer/random_uniformAdd5up_level_1_no_0/kernel/Initializer/random_uniform/mul5up_level_1_no_0/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@up_level_1_no_0/kernel*'
_output_shapes
:?@
?
up_level_1_no_0/kernelVarHandleOp*)
_class
loc:@up_level_1_no_0/kernel*
_output_shapes
: *
	container *
dtype0*
shape:?@*'
shared_nameup_level_1_no_0/kernel
}
7up_level_1_no_0/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpup_level_1_no_0/kernel*
_output_shapes
: 
?
up_level_1_no_0/kernel/AssignAssignVariableOpup_level_1_no_0/kernel1up_level_1_no_0/kernel/Initializer/random_uniform*
dtype0
?
*up_level_1_no_0/kernel/Read/ReadVariableOpReadVariableOpup_level_1_no_0/kernel*'
_output_shapes
:?@*
dtype0
?
&up_level_1_no_0/bias/Initializer/zerosConst*'
_class
loc:@up_level_1_no_0/bias*
_output_shapes
:@*
dtype0*
valueB@*    
?
up_level_1_no_0/biasVarHandleOp*'
_class
loc:@up_level_1_no_0/bias*
_output_shapes
: *
	container *
dtype0*
shape:@*%
shared_nameup_level_1_no_0/bias
y
5up_level_1_no_0/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpup_level_1_no_0/bias*
_output_shapes
: 
z
up_level_1_no_0/bias/AssignAssignVariableOpup_level_1_no_0/bias&up_level_1_no_0/bias/Initializer/zeros*
dtype0
y
(up_level_1_no_0/bias/Read/ReadVariableOpReadVariableOpup_level_1_no_0/bias*
_output_shapes
:@*
dtype0
n
up_level_1_no_0/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
?
%up_level_1_no_0/Conv2D/ReadVariableOpReadVariableOpup_level_1_no_0/kernel*'
_output_shapes
:?@*
dtype0
?
up_level_1_no_0/Conv2DConv2Dconcatenate_7/concat%up_level_1_no_0/Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
w
&up_level_1_no_0/BiasAdd/ReadVariableOpReadVariableOpup_level_1_no_0/bias*
_output_shapes
:@*
dtype0
?
up_level_1_no_0/BiasAddBiasAddup_level_1_no_0/Conv2D&up_level_1_no_0/BiasAdd/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@*
data_formatNHWC
?
up_level_1_no_0/ReluReluup_level_1_no_0/BiasAdd*
T0*A
_output_shapes/
-:+???????????????????????????@
?
7up_level_1_no_2/kernel/Initializer/random_uniform/shapeConst*)
_class
loc:@up_level_1_no_2/kernel*
_output_shapes
:*
dtype0*%
valueB"      @       
?
5up_level_1_no_2/kernel/Initializer/random_uniform/minConst*)
_class
loc:@up_level_1_no_2/kernel*
_output_shapes
: *
dtype0*
valueB
 *????
?
5up_level_1_no_2/kernel/Initializer/random_uniform/maxConst*)
_class
loc:@up_level_1_no_2/kernel*
_output_shapes
: *
dtype0*
valueB
 *???=
?
?up_level_1_no_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform7up_level_1_no_2/kernel/Initializer/random_uniform/shape*
T0*)
_class
loc:@up_level_1_no_2/kernel*&
_output_shapes
:@ *
dtype0*

seed *
seed2 
?
5up_level_1_no_2/kernel/Initializer/random_uniform/subSub5up_level_1_no_2/kernel/Initializer/random_uniform/max5up_level_1_no_2/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@up_level_1_no_2/kernel*
_output_shapes
: 
?
5up_level_1_no_2/kernel/Initializer/random_uniform/mulMul?up_level_1_no_2/kernel/Initializer/random_uniform/RandomUniform5up_level_1_no_2/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@up_level_1_no_2/kernel*&
_output_shapes
:@ 
?
1up_level_1_no_2/kernel/Initializer/random_uniformAdd5up_level_1_no_2/kernel/Initializer/random_uniform/mul5up_level_1_no_2/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@up_level_1_no_2/kernel*&
_output_shapes
:@ 
?
up_level_1_no_2/kernelVarHandleOp*)
_class
loc:@up_level_1_no_2/kernel*
_output_shapes
: *
	container *
dtype0*
shape:@ *'
shared_nameup_level_1_no_2/kernel
}
7up_level_1_no_2/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpup_level_1_no_2/kernel*
_output_shapes
: 
?
up_level_1_no_2/kernel/AssignAssignVariableOpup_level_1_no_2/kernel1up_level_1_no_2/kernel/Initializer/random_uniform*
dtype0
?
*up_level_1_no_2/kernel/Read/ReadVariableOpReadVariableOpup_level_1_no_2/kernel*&
_output_shapes
:@ *
dtype0
?
&up_level_1_no_2/bias/Initializer/zerosConst*'
_class
loc:@up_level_1_no_2/bias*
_output_shapes
: *
dtype0*
valueB *    
?
up_level_1_no_2/biasVarHandleOp*'
_class
loc:@up_level_1_no_2/bias*
_output_shapes
: *
	container *
dtype0*
shape: *%
shared_nameup_level_1_no_2/bias
y
5up_level_1_no_2/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpup_level_1_no_2/bias*
_output_shapes
: 
z
up_level_1_no_2/bias/AssignAssignVariableOpup_level_1_no_2/bias&up_level_1_no_2/bias/Initializer/zeros*
dtype0
y
(up_level_1_no_2/bias/Read/ReadVariableOpReadVariableOpup_level_1_no_2/bias*
_output_shapes
: *
dtype0
n
up_level_1_no_2/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
?
%up_level_1_no_2/Conv2D/ReadVariableOpReadVariableOpup_level_1_no_2/kernel*&
_output_shapes
:@ *
dtype0
?
up_level_1_no_2/Conv2DConv2Dup_level_1_no_0/Relu%up_level_1_no_2/Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? *
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
w
&up_level_1_no_2/BiasAdd/ReadVariableOpReadVariableOpup_level_1_no_2/bias*
_output_shapes
: *
dtype0
?
up_level_1_no_2/BiasAddBiasAddup_level_1_no_2/Conv2D&up_level_1_no_2/BiasAdd/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? *
data_formatNHWC
?
up_level_1_no_2/ReluReluup_level_1_no_2/BiasAdd*
T0*A
_output_shapes/
-:+??????????????????????????? 
i
up_sampling2d_8/ShapeShapeup_level_1_no_2/Relu*
T0*
_output_shapes
:*
out_type0
m
#up_sampling2d_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
o
%up_sampling2d_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
o
%up_sampling2d_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
up_sampling2d_8/strided_sliceStridedSliceup_sampling2d_8/Shape#up_sampling2d_8/strided_slice/stack%up_sampling2d_8/strided_slice/stack_1%up_sampling2d_8/strided_slice/stack_2*
Index0*
T0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask 
f
up_sampling2d_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"      
u
up_sampling2d_8/mulMulup_sampling2d_8/strided_sliceup_sampling2d_8/Const*
T0*
_output_shapes
:
?
,up_sampling2d_8/resize/ResizeNearestNeighborResizeNearestNeighborup_level_1_no_2/Reluup_sampling2d_8/mul*
T0*A
_output_shapes/
-:+??????????????????????????? *
align_corners( *
half_pixel_centers(
[
concatenate_8/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
?
concatenate_8/concatConcatV2,up_sampling2d_8/resize/ResizeNearestNeighbordown_level_0_no_1/Reluconcatenate_8/concat/axis*
N*
T0*

Tidx0*A
_output_shapes/
-:+???????????????????????????@
?
7up_level_0_no_0/kernel/Initializer/random_uniform/shapeConst*)
_class
loc:@up_level_0_no_0/kernel*
_output_shapes
:*
dtype0*%
valueB"      @       
?
5up_level_0_no_0/kernel/Initializer/random_uniform/minConst*)
_class
loc:@up_level_0_no_0/kernel*
_output_shapes
: *
dtype0*
valueB
 *????
?
5up_level_0_no_0/kernel/Initializer/random_uniform/maxConst*)
_class
loc:@up_level_0_no_0/kernel*
_output_shapes
: *
dtype0*
valueB
 *???=
?
?up_level_0_no_0/kernel/Initializer/random_uniform/RandomUniformRandomUniform7up_level_0_no_0/kernel/Initializer/random_uniform/shape*
T0*)
_class
loc:@up_level_0_no_0/kernel*&
_output_shapes
:@ *
dtype0*

seed *
seed2 
?
5up_level_0_no_0/kernel/Initializer/random_uniform/subSub5up_level_0_no_0/kernel/Initializer/random_uniform/max5up_level_0_no_0/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@up_level_0_no_0/kernel*
_output_shapes
: 
?
5up_level_0_no_0/kernel/Initializer/random_uniform/mulMul?up_level_0_no_0/kernel/Initializer/random_uniform/RandomUniform5up_level_0_no_0/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@up_level_0_no_0/kernel*&
_output_shapes
:@ 
?
1up_level_0_no_0/kernel/Initializer/random_uniformAdd5up_level_0_no_0/kernel/Initializer/random_uniform/mul5up_level_0_no_0/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@up_level_0_no_0/kernel*&
_output_shapes
:@ 
?
up_level_0_no_0/kernelVarHandleOp*)
_class
loc:@up_level_0_no_0/kernel*
_output_shapes
: *
	container *
dtype0*
shape:@ *'
shared_nameup_level_0_no_0/kernel
}
7up_level_0_no_0/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpup_level_0_no_0/kernel*
_output_shapes
: 
?
up_level_0_no_0/kernel/AssignAssignVariableOpup_level_0_no_0/kernel1up_level_0_no_0/kernel/Initializer/random_uniform*
dtype0
?
*up_level_0_no_0/kernel/Read/ReadVariableOpReadVariableOpup_level_0_no_0/kernel*&
_output_shapes
:@ *
dtype0
?
&up_level_0_no_0/bias/Initializer/zerosConst*'
_class
loc:@up_level_0_no_0/bias*
_output_shapes
: *
dtype0*
valueB *    
?
up_level_0_no_0/biasVarHandleOp*'
_class
loc:@up_level_0_no_0/bias*
_output_shapes
: *
	container *
dtype0*
shape: *%
shared_nameup_level_0_no_0/bias
y
5up_level_0_no_0/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpup_level_0_no_0/bias*
_output_shapes
: 
z
up_level_0_no_0/bias/AssignAssignVariableOpup_level_0_no_0/bias&up_level_0_no_0/bias/Initializer/zeros*
dtype0
y
(up_level_0_no_0/bias/Read/ReadVariableOpReadVariableOpup_level_0_no_0/bias*
_output_shapes
: *
dtype0
n
up_level_0_no_0/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
?
%up_level_0_no_0/Conv2D/ReadVariableOpReadVariableOpup_level_0_no_0/kernel*&
_output_shapes
:@ *
dtype0
?
up_level_0_no_0/Conv2DConv2Dconcatenate_8/concat%up_level_0_no_0/Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? *
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
w
&up_level_0_no_0/BiasAdd/ReadVariableOpReadVariableOpup_level_0_no_0/bias*
_output_shapes
: *
dtype0
?
up_level_0_no_0/BiasAddBiasAddup_level_0_no_0/Conv2D&up_level_0_no_0/BiasAdd/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? *
data_formatNHWC
?
up_level_0_no_0/ReluReluup_level_0_no_0/BiasAdd*
T0*A
_output_shapes/
-:+??????????????????????????? 
?
7up_level_0_no_2/kernel/Initializer/random_uniform/shapeConst*)
_class
loc:@up_level_0_no_2/kernel*
_output_shapes
:*
dtype0*%
valueB"              
?
5up_level_0_no_2/kernel/Initializer/random_uniform/minConst*)
_class
loc:@up_level_0_no_2/kernel*
_output_shapes
: *
dtype0*
valueB
 *?ѽ
?
5up_level_0_no_2/kernel/Initializer/random_uniform/maxConst*)
_class
loc:@up_level_0_no_2/kernel*
_output_shapes
: *
dtype0*
valueB
 *??=
?
?up_level_0_no_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform7up_level_0_no_2/kernel/Initializer/random_uniform/shape*
T0*)
_class
loc:@up_level_0_no_2/kernel*&
_output_shapes
:  *
dtype0*

seed *
seed2 
?
5up_level_0_no_2/kernel/Initializer/random_uniform/subSub5up_level_0_no_2/kernel/Initializer/random_uniform/max5up_level_0_no_2/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@up_level_0_no_2/kernel*
_output_shapes
: 
?
5up_level_0_no_2/kernel/Initializer/random_uniform/mulMul?up_level_0_no_2/kernel/Initializer/random_uniform/RandomUniform5up_level_0_no_2/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@up_level_0_no_2/kernel*&
_output_shapes
:  
?
1up_level_0_no_2/kernel/Initializer/random_uniformAdd5up_level_0_no_2/kernel/Initializer/random_uniform/mul5up_level_0_no_2/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@up_level_0_no_2/kernel*&
_output_shapes
:  
?
up_level_0_no_2/kernelVarHandleOp*)
_class
loc:@up_level_0_no_2/kernel*
_output_shapes
: *
	container *
dtype0*
shape:  *'
shared_nameup_level_0_no_2/kernel
}
7up_level_0_no_2/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpup_level_0_no_2/kernel*
_output_shapes
: 
?
up_level_0_no_2/kernel/AssignAssignVariableOpup_level_0_no_2/kernel1up_level_0_no_2/kernel/Initializer/random_uniform*
dtype0
?
*up_level_0_no_2/kernel/Read/ReadVariableOpReadVariableOpup_level_0_no_2/kernel*&
_output_shapes
:  *
dtype0
?
&up_level_0_no_2/bias/Initializer/zerosConst*'
_class
loc:@up_level_0_no_2/bias*
_output_shapes
: *
dtype0*
valueB *    
?
up_level_0_no_2/biasVarHandleOp*'
_class
loc:@up_level_0_no_2/bias*
_output_shapes
: *
	container *
dtype0*
shape: *%
shared_nameup_level_0_no_2/bias
y
5up_level_0_no_2/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpup_level_0_no_2/bias*
_output_shapes
: 
z
up_level_0_no_2/bias/AssignAssignVariableOpup_level_0_no_2/bias&up_level_0_no_2/bias/Initializer/zeros*
dtype0
y
(up_level_0_no_2/bias/Read/ReadVariableOpReadVariableOpup_level_0_no_2/bias*
_output_shapes
: *
dtype0
n
up_level_0_no_2/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
?
%up_level_0_no_2/Conv2D/ReadVariableOpReadVariableOpup_level_0_no_2/kernel*&
_output_shapes
:  *
dtype0
?
up_level_0_no_2/Conv2DConv2Dup_level_0_no_0/Relu%up_level_0_no_2/Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? *
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
w
&up_level_0_no_2/BiasAdd/ReadVariableOpReadVariableOpup_level_0_no_2/bias*
_output_shapes
: *
dtype0
?
up_level_0_no_2/BiasAddBiasAddup_level_0_no_2/Conv2D&up_level_0_no_2/BiasAdd/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? *
data_formatNHWC
?
up_level_0_no_2/ReluReluup_level_0_no_2/BiasAdd*
T0*A
_output_shapes/
-:+??????????????????????????? 
?
0features/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@features/kernel*
_output_shapes
:*
dtype0*%
valueB"          ?   
?
.features/kernel/Initializer/random_uniform/minConst*"
_class
loc:@features/kernel*
_output_shapes
: *
dtype0*
valueB
 *?2??
?
.features/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@features/kernel*
_output_shapes
: *
dtype0*
valueB
 *?2?=
?
8features/kernel/Initializer/random_uniform/RandomUniformRandomUniform0features/kernel/Initializer/random_uniform/shape*
T0*"
_class
loc:@features/kernel*'
_output_shapes
: ?*
dtype0*

seed *
seed2 
?
.features/kernel/Initializer/random_uniform/subSub.features/kernel/Initializer/random_uniform/max.features/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@features/kernel*
_output_shapes
: 
?
.features/kernel/Initializer/random_uniform/mulMul8features/kernel/Initializer/random_uniform/RandomUniform.features/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@features/kernel*'
_output_shapes
: ?
?
*features/kernel/Initializer/random_uniformAdd.features/kernel/Initializer/random_uniform/mul.features/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@features/kernel*'
_output_shapes
: ?
?
features/kernelVarHandleOp*"
_class
loc:@features/kernel*
_output_shapes
: *
	container *
dtype0*
shape: ?* 
shared_namefeatures/kernel
o
0features/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpfeatures/kernel*
_output_shapes
: 
t
features/kernel/AssignAssignVariableOpfeatures/kernel*features/kernel/Initializer/random_uniform*
dtype0
|
#features/kernel/Read/ReadVariableOpReadVariableOpfeatures/kernel*'
_output_shapes
: ?*
dtype0
?
features/bias/Initializer/zerosConst* 
_class
loc:@features/bias*
_output_shapes	
:?*
dtype0*
valueB?*    
?
features/biasVarHandleOp* 
_class
loc:@features/bias*
_output_shapes
: *
	container *
dtype0*
shape:?*
shared_namefeatures/bias
k
.features/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpfeatures/bias*
_output_shapes
: 
e
features/bias/AssignAssignVariableOpfeatures/biasfeatures/bias/Initializer/zeros*
dtype0
l
!features/bias/Read/ReadVariableOpReadVariableOpfeatures/bias*
_output_shapes	
:?*
dtype0
g
features/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
w
features/Conv2D/ReadVariableOpReadVariableOpfeatures/kernel*'
_output_shapes
: ?*
dtype0
?
features/Conv2DConv2Dup_level_0_no_2/Relufeatures/Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
j
features/BiasAdd/ReadVariableOpReadVariableOpfeatures/bias*
_output_shapes	
:?*
dtype0
?
features/BiasAddBiasAddfeatures/Conv2Dfeatures/BiasAdd/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????*
data_formatNHWC
t
features/ReluRelufeatures/BiasAdd*
T0*B
_output_shapes0
.:,????????????????????????????
?
,prob/kernel/Initializer/random_uniform/shapeConst*
_class
loc:@prob/kernel*
_output_shapes
:*
dtype0*%
valueB"      ?      
?
*prob/kernel/Initializer/random_uniform/minConst*
_class
loc:@prob/kernel*
_output_shapes
: *
dtype0*
valueB
 *n?\?
?
*prob/kernel/Initializer/random_uniform/maxConst*
_class
loc:@prob/kernel*
_output_shapes
: *
dtype0*
valueB
 *n?\>
?
4prob/kernel/Initializer/random_uniform/RandomUniformRandomUniform,prob/kernel/Initializer/random_uniform/shape*
T0*
_class
loc:@prob/kernel*'
_output_shapes
:?*
dtype0*

seed *
seed2 
?
*prob/kernel/Initializer/random_uniform/subSub*prob/kernel/Initializer/random_uniform/max*prob/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@prob/kernel*
_output_shapes
: 
?
*prob/kernel/Initializer/random_uniform/mulMul4prob/kernel/Initializer/random_uniform/RandomUniform*prob/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@prob/kernel*'
_output_shapes
:?
?
&prob/kernel/Initializer/random_uniformAdd*prob/kernel/Initializer/random_uniform/mul*prob/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@prob/kernel*'
_output_shapes
:?
?
prob/kernelVarHandleOp*
_class
loc:@prob/kernel*
_output_shapes
: *
	container *
dtype0*
shape:?*
shared_nameprob/kernel
g
,prob/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpprob/kernel*
_output_shapes
: 
h
prob/kernel/AssignAssignVariableOpprob/kernel&prob/kernel/Initializer/random_uniform*
dtype0
t
prob/kernel/Read/ReadVariableOpReadVariableOpprob/kernel*'
_output_shapes
:?*
dtype0
?
prob/bias/Initializer/zerosConst*
_class
loc:@prob/bias*
_output_shapes
:*
dtype0*
valueB*    
?
	prob/biasVarHandleOp*
_class
loc:@prob/bias*
_output_shapes
: *
	container *
dtype0*
shape:*
shared_name	prob/bias
c
*prob/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp	prob/bias*
_output_shapes
: 
Y
prob/bias/AssignAssignVariableOp	prob/biasprob/bias/Initializer/zeros*
dtype0
c
prob/bias/Read/ReadVariableOpReadVariableOp	prob/bias*
_output_shapes
:*
dtype0
c
prob/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
o
prob/Conv2D/ReadVariableOpReadVariableOpprob/kernel*'
_output_shapes
:?*
dtype0
?
prob/Conv2DConv2Dfeatures/Reluprob/Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
a
prob/BiasAdd/ReadVariableOpReadVariableOp	prob/bias*
_output_shapes
:*
dtype0
?
prob/BiasAddBiasAddprob/Conv2Dprob/BiasAdd/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????*
data_formatNHWC
q
prob/SigmoidSigmoidprob/BiasAdd*
T0*A
_output_shapes/
-:+???????????????????????????
?
,dist/kernel/Initializer/random_uniform/shapeConst*
_class
loc:@dist/kernel*
_output_shapes
:*
dtype0*%
valueB"      ?       
?
*dist/kernel/Initializer/random_uniform/minConst*
_class
loc:@dist/kernel*
_output_shapes
: *
dtype0*
valueB
 *?KF?
?
*dist/kernel/Initializer/random_uniform/maxConst*
_class
loc:@dist/kernel*
_output_shapes
: *
dtype0*
valueB
 *?KF>
?
4dist/kernel/Initializer/random_uniform/RandomUniformRandomUniform,dist/kernel/Initializer/random_uniform/shape*
T0*
_class
loc:@dist/kernel*'
_output_shapes
:? *
dtype0*

seed *
seed2 
?
*dist/kernel/Initializer/random_uniform/subSub*dist/kernel/Initializer/random_uniform/max*dist/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dist/kernel*
_output_shapes
: 
?
*dist/kernel/Initializer/random_uniform/mulMul4dist/kernel/Initializer/random_uniform/RandomUniform*dist/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@dist/kernel*'
_output_shapes
:? 
?
&dist/kernel/Initializer/random_uniformAdd*dist/kernel/Initializer/random_uniform/mul*dist/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dist/kernel*'
_output_shapes
:? 
?
dist/kernelVarHandleOp*
_class
loc:@dist/kernel*
_output_shapes
: *
	container *
dtype0*
shape:? *
shared_namedist/kernel
g
,dist/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdist/kernel*
_output_shapes
: 
h
dist/kernel/AssignAssignVariableOpdist/kernel&dist/kernel/Initializer/random_uniform*
dtype0
t
dist/kernel/Read/ReadVariableOpReadVariableOpdist/kernel*'
_output_shapes
:? *
dtype0
?
dist/bias/Initializer/zerosConst*
_class
loc:@dist/bias*
_output_shapes
: *
dtype0*
valueB *    
?
	dist/biasVarHandleOp*
_class
loc:@dist/bias*
_output_shapes
: *
	container *
dtype0*
shape: *
shared_name	dist/bias
c
*dist/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp	dist/bias*
_output_shapes
: 
Y
dist/bias/AssignAssignVariableOp	dist/biasdist/bias/Initializer/zeros*
dtype0
c
dist/bias/Read/ReadVariableOpReadVariableOp	dist/bias*
_output_shapes
: *
dtype0
c
dist/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
o
dist/Conv2D/ReadVariableOpReadVariableOpdist/kernel*'
_output_shapes
:? *
dtype0
?
dist/Conv2DConv2Dfeatures/Reludist/Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? *
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
a
dist/BiasAdd/ReadVariableOpReadVariableOp	dist/bias*
_output_shapes
: *
dtype0
?
dist/BiasAddBiasAdddist/Conv2Ddist/BiasAdd/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? *
data_formatNHWC
?
*conv2d_transpose_1/kernel/Initializer/onesConst*,
_class"
 loc:@conv2d_transpose_1/kernel*&
_output_shapes
:*
dtype0*%
valueB*  ??
?
conv2d_transpose_1/kernelVarHandleOp*,
_class"
 loc:@conv2d_transpose_1/kernel*
_output_shapes
: *
	container *
dtype0*
shape:**
shared_nameconv2d_transpose_1/kernel
?
:conv2d_transpose_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_transpose_1/kernel*
_output_shapes
: 
?
 conv2d_transpose_1/kernel/AssignAssignVariableOpconv2d_transpose_1/kernel*conv2d_transpose_1/kernel/Initializer/ones*
dtype0
?
-conv2d_transpose_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/kernel*&
_output_shapes
:*
dtype0
d
conv2d_transpose_1/ShapeShapeprob/Sigmoid*
T0*
_output_shapes
:*
out_type0
p
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
r
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
r
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
 conv2d_transpose_1/strided_sliceStridedSliceconv2d_transpose_1/Shape&conv2d_transpose_1/strided_slice/stack(conv2d_transpose_1/strided_slice/stack_1(conv2d_transpose_1/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
r
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
t
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
t
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
"conv2d_transpose_1/strided_slice_1StridedSliceconv2d_transpose_1/Shape(conv2d_transpose_1/strided_slice_1/stack*conv2d_transpose_1/strided_slice_1/stack_1*conv2d_transpose_1/strided_slice_1/stack_2*
Index0*
T0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
r
(conv2d_transpose_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
t
*conv2d_transpose_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
t
*conv2d_transpose_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
"conv2d_transpose_1/strided_slice_2StridedSliceconv2d_transpose_1/Shape(conv2d_transpose_1/strided_slice_2/stack*conv2d_transpose_1/strided_slice_2/stack_1*conv2d_transpose_1/strided_slice_2/stack_2*
Index0*
T0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
Z
conv2d_transpose_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
|
conv2d_transpose_1/mulMul"conv2d_transpose_1/strided_slice_1conv2d_transpose_1/mul/y*
T0*
_output_shapes
: 
\
conv2d_transpose_1/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :
?
conv2d_transpose_1/mul_1Mul"conv2d_transpose_1/strided_slice_2conv2d_transpose_1/mul_1/y*
T0*
_output_shapes
: 
\
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :
?
conv2d_transpose_1/stackPack conv2d_transpose_1/strided_sliceconv2d_transpose_1/mulconv2d_transpose_1/mul_1conv2d_transpose_1/stack/3*
N*
T0*
_output_shapes
:*

axis 
r
(conv2d_transpose_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 
t
*conv2d_transpose_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
t
*conv2d_transpose_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
"conv2d_transpose_1/strided_slice_3StridedSliceconv2d_transpose_1/stack(conv2d_transpose_1/strided_slice_3/stack*conv2d_transpose_1/strided_slice_3/stack_1*conv2d_transpose_1/strided_slice_3/stack_2*
Index0*
T0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
?
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpconv2d_transpose_1/kernel*&
_output_shapes
:*
dtype0
?
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInputconv2d_transpose_1/stack2conv2d_transpose_1/conv2d_transpose/ReadVariableOpprob/Sigmoid*
T0*A
_output_shapes/
-:+???????????????????????????*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
b
up_sampling2d_10/ShapeShapedist/BiasAdd*
T0*
_output_shapes
:*
out_type0
n
$up_sampling2d_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
p
&up_sampling2d_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
p
&up_sampling2d_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
up_sampling2d_10/strided_sliceStridedSliceup_sampling2d_10/Shape$up_sampling2d_10/strided_slice/stack&up_sampling2d_10/strided_slice/stack_1&up_sampling2d_10/strided_slice/stack_2*
Index0*
T0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask 
g
up_sampling2d_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"      
x
up_sampling2d_10/mulMulup_sampling2d_10/strided_sliceup_sampling2d_10/Const*
T0*
_output_shapes
:
?
-up_sampling2d_10/resize/ResizeNearestNeighborResizeNearestNeighbordist/BiasAddup_sampling2d_10/mul*
T0*A
_output_shapes/
-:+??????????????????????????? *
align_corners( *
half_pixel_centers(
\
concatenate_10/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
?
concatenate_10/concatConcatV2#conv2d_transpose_1/conv2d_transpose-up_sampling2d_10/resize/ResizeNearestNeighborconcatenate_10/concat/axis*
N*
T0*

Tidx0*A
_output_shapes/
-:+???????????????????????????!
?
PlaceholderPlaceholder*J
_output_shapes8
6:4????????????????????????????????????*
dtype0*?
shape6:4????????????????????????????????????
O
AssignVariableOpAssignVariableOpconv2d_4/kernelPlaceholder*
dtype0
y
ReadVariableOpReadVariableOpconv2d_4/kernel^AssignVariableOp*&
_output_shapes
: *
dtype0
h
Placeholder_1Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
Q
AssignVariableOp_1AssignVariableOpconv2d_4/biasPlaceholder_1*
dtype0
o
ReadVariableOp_1ReadVariableOpconv2d_4/bias^AssignVariableOp_1*
_output_shapes
: *
dtype0
?
Placeholder_2Placeholder*J
_output_shapes8
6:4????????????????????????????????????*
dtype0*?
shape6:4????????????????????????????????????
S
AssignVariableOp_2AssignVariableOpconv2d_5/kernelPlaceholder_2*
dtype0
}
ReadVariableOp_2ReadVariableOpconv2d_5/kernel^AssignVariableOp_2*&
_output_shapes
:  *
dtype0
h
Placeholder_3Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
Q
AssignVariableOp_3AssignVariableOpconv2d_5/biasPlaceholder_3*
dtype0
o
ReadVariableOp_3ReadVariableOpconv2d_5/bias^AssignVariableOp_3*
_output_shapes
: *
dtype0
?
Placeholder_4Placeholder*J
_output_shapes8
6:4????????????????????????????????????*
dtype0*?
shape6:4????????????????????????????????????
\
AssignVariableOp_4AssignVariableOpdown_level_0_no_0/kernelPlaceholder_4*
dtype0
?
ReadVariableOp_4ReadVariableOpdown_level_0_no_0/kernel^AssignVariableOp_4*&
_output_shapes
:  *
dtype0
h
Placeholder_5Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
Z
AssignVariableOp_5AssignVariableOpdown_level_0_no_0/biasPlaceholder_5*
dtype0
x
ReadVariableOp_5ReadVariableOpdown_level_0_no_0/bias^AssignVariableOp_5*
_output_shapes
: *
dtype0
?
Placeholder_6Placeholder*J
_output_shapes8
6:4????????????????????????????????????*
dtype0*?
shape6:4????????????????????????????????????
\
AssignVariableOp_6AssignVariableOpdown_level_0_no_1/kernelPlaceholder_6*
dtype0
?
ReadVariableOp_6ReadVariableOpdown_level_0_no_1/kernel^AssignVariableOp_6*&
_output_shapes
:  *
dtype0
h
Placeholder_7Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
Z
AssignVariableOp_7AssignVariableOpdown_level_0_no_1/biasPlaceholder_7*
dtype0
x
ReadVariableOp_7ReadVariableOpdown_level_0_no_1/bias^AssignVariableOp_7*
_output_shapes
: *
dtype0
?
Placeholder_8Placeholder*J
_output_shapes8
6:4????????????????????????????????????*
dtype0*?
shape6:4????????????????????????????????????
\
AssignVariableOp_8AssignVariableOpdown_level_1_no_0/kernelPlaceholder_8*
dtype0
?
ReadVariableOp_8ReadVariableOpdown_level_1_no_0/kernel^AssignVariableOp_8*&
_output_shapes
: @*
dtype0
h
Placeholder_9Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
Z
AssignVariableOp_9AssignVariableOpdown_level_1_no_0/biasPlaceholder_9*
dtype0
x
ReadVariableOp_9ReadVariableOpdown_level_1_no_0/bias^AssignVariableOp_9*
_output_shapes
:@*
dtype0
?
Placeholder_10Placeholder*J
_output_shapes8
6:4????????????????????????????????????*
dtype0*?
shape6:4????????????????????????????????????
^
AssignVariableOp_10AssignVariableOpdown_level_1_no_1/kernelPlaceholder_10*
dtype0
?
ReadVariableOp_10ReadVariableOpdown_level_1_no_1/kernel^AssignVariableOp_10*&
_output_shapes
:@@*
dtype0
i
Placeholder_11Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
\
AssignVariableOp_11AssignVariableOpdown_level_1_no_1/biasPlaceholder_11*
dtype0
z
ReadVariableOp_11ReadVariableOpdown_level_1_no_1/bias^AssignVariableOp_11*
_output_shapes
:@*
dtype0
?
Placeholder_12Placeholder*J
_output_shapes8
6:4????????????????????????????????????*
dtype0*?
shape6:4????????????????????????????????????
^
AssignVariableOp_12AssignVariableOpdown_level_2_no_0/kernelPlaceholder_12*
dtype0
?
ReadVariableOp_12ReadVariableOpdown_level_2_no_0/kernel^AssignVariableOp_12*'
_output_shapes
:@?*
dtype0
i
Placeholder_13Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
\
AssignVariableOp_13AssignVariableOpdown_level_2_no_0/biasPlaceholder_13*
dtype0
{
ReadVariableOp_13ReadVariableOpdown_level_2_no_0/bias^AssignVariableOp_13*
_output_shapes	
:?*
dtype0
?
Placeholder_14Placeholder*J
_output_shapes8
6:4????????????????????????????????????*
dtype0*?
shape6:4????????????????????????????????????
^
AssignVariableOp_14AssignVariableOpdown_level_2_no_1/kernelPlaceholder_14*
dtype0
?
ReadVariableOp_14ReadVariableOpdown_level_2_no_1/kernel^AssignVariableOp_14*(
_output_shapes
:??*
dtype0
i
Placeholder_15Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
\
AssignVariableOp_15AssignVariableOpdown_level_2_no_1/biasPlaceholder_15*
dtype0
{
ReadVariableOp_15ReadVariableOpdown_level_2_no_1/bias^AssignVariableOp_15*
_output_shapes	
:?*
dtype0
?
Placeholder_16Placeholder*J
_output_shapes8
6:4????????????????????????????????????*
dtype0*?
shape6:4????????????????????????????????????
U
AssignVariableOp_16AssignVariableOpmiddle_0/kernelPlaceholder_16*
dtype0
?
ReadVariableOp_16ReadVariableOpmiddle_0/kernel^AssignVariableOp_16*(
_output_shapes
:??*
dtype0
i
Placeholder_17Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
S
AssignVariableOp_17AssignVariableOpmiddle_0/biasPlaceholder_17*
dtype0
r
ReadVariableOp_17ReadVariableOpmiddle_0/bias^AssignVariableOp_17*
_output_shapes	
:?*
dtype0
?
Placeholder_18Placeholder*J
_output_shapes8
6:4????????????????????????????????????*
dtype0*?
shape6:4????????????????????????????????????
U
AssignVariableOp_18AssignVariableOpmiddle_2/kernelPlaceholder_18*
dtype0
?
ReadVariableOp_18ReadVariableOpmiddle_2/kernel^AssignVariableOp_18*(
_output_shapes
:??*
dtype0
i
Placeholder_19Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
S
AssignVariableOp_19AssignVariableOpmiddle_2/biasPlaceholder_19*
dtype0
r
ReadVariableOp_19ReadVariableOpmiddle_2/bias^AssignVariableOp_19*
_output_shapes	
:?*
dtype0
?
Placeholder_20Placeholder*J
_output_shapes8
6:4????????????????????????????????????*
dtype0*?
shape6:4????????????????????????????????????
\
AssignVariableOp_20AssignVariableOpup_level_2_no_0/kernelPlaceholder_20*
dtype0
?
ReadVariableOp_20ReadVariableOpup_level_2_no_0/kernel^AssignVariableOp_20*(
_output_shapes
:??*
dtype0
i
Placeholder_21Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
Z
AssignVariableOp_21AssignVariableOpup_level_2_no_0/biasPlaceholder_21*
dtype0
y
ReadVariableOp_21ReadVariableOpup_level_2_no_0/bias^AssignVariableOp_21*
_output_shapes	
:?*
dtype0
?
Placeholder_22Placeholder*J
_output_shapes8
6:4????????????????????????????????????*
dtype0*?
shape6:4????????????????????????????????????
\
AssignVariableOp_22AssignVariableOpup_level_2_no_2/kernelPlaceholder_22*
dtype0
?
ReadVariableOp_22ReadVariableOpup_level_2_no_2/kernel^AssignVariableOp_22*'
_output_shapes
:?@*
dtype0
i
Placeholder_23Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
Z
AssignVariableOp_23AssignVariableOpup_level_2_no_2/biasPlaceholder_23*
dtype0
x
ReadVariableOp_23ReadVariableOpup_level_2_no_2/bias^AssignVariableOp_23*
_output_shapes
:@*
dtype0
?
Placeholder_24Placeholder*J
_output_shapes8
6:4????????????????????????????????????*
dtype0*?
shape6:4????????????????????????????????????
\
AssignVariableOp_24AssignVariableOpup_level_1_no_0/kernelPlaceholder_24*
dtype0
?
ReadVariableOp_24ReadVariableOpup_level_1_no_0/kernel^AssignVariableOp_24*'
_output_shapes
:?@*
dtype0
i
Placeholder_25Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
Z
AssignVariableOp_25AssignVariableOpup_level_1_no_0/biasPlaceholder_25*
dtype0
x
ReadVariableOp_25ReadVariableOpup_level_1_no_0/bias^AssignVariableOp_25*
_output_shapes
:@*
dtype0
?
Placeholder_26Placeholder*J
_output_shapes8
6:4????????????????????????????????????*
dtype0*?
shape6:4????????????????????????????????????
\
AssignVariableOp_26AssignVariableOpup_level_1_no_2/kernelPlaceholder_26*
dtype0
?
ReadVariableOp_26ReadVariableOpup_level_1_no_2/kernel^AssignVariableOp_26*&
_output_shapes
:@ *
dtype0
i
Placeholder_27Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
Z
AssignVariableOp_27AssignVariableOpup_level_1_no_2/biasPlaceholder_27*
dtype0
x
ReadVariableOp_27ReadVariableOpup_level_1_no_2/bias^AssignVariableOp_27*
_output_shapes
: *
dtype0
?
Placeholder_28Placeholder*J
_output_shapes8
6:4????????????????????????????????????*
dtype0*?
shape6:4????????????????????????????????????
\
AssignVariableOp_28AssignVariableOpup_level_0_no_0/kernelPlaceholder_28*
dtype0
?
ReadVariableOp_28ReadVariableOpup_level_0_no_0/kernel^AssignVariableOp_28*&
_output_shapes
:@ *
dtype0
i
Placeholder_29Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
Z
AssignVariableOp_29AssignVariableOpup_level_0_no_0/biasPlaceholder_29*
dtype0
x
ReadVariableOp_29ReadVariableOpup_level_0_no_0/bias^AssignVariableOp_29*
_output_shapes
: *
dtype0
?
Placeholder_30Placeholder*J
_output_shapes8
6:4????????????????????????????????????*
dtype0*?
shape6:4????????????????????????????????????
\
AssignVariableOp_30AssignVariableOpup_level_0_no_2/kernelPlaceholder_30*
dtype0
?
ReadVariableOp_30ReadVariableOpup_level_0_no_2/kernel^AssignVariableOp_30*&
_output_shapes
:  *
dtype0
i
Placeholder_31Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
Z
AssignVariableOp_31AssignVariableOpup_level_0_no_2/biasPlaceholder_31*
dtype0
x
ReadVariableOp_31ReadVariableOpup_level_0_no_2/bias^AssignVariableOp_31*
_output_shapes
: *
dtype0
?
Placeholder_32Placeholder*J
_output_shapes8
6:4????????????????????????????????????*
dtype0*?
shape6:4????????????????????????????????????
U
AssignVariableOp_32AssignVariableOpfeatures/kernelPlaceholder_32*
dtype0
?
ReadVariableOp_32ReadVariableOpfeatures/kernel^AssignVariableOp_32*'
_output_shapes
: ?*
dtype0
i
Placeholder_33Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
S
AssignVariableOp_33AssignVariableOpfeatures/biasPlaceholder_33*
dtype0
r
ReadVariableOp_33ReadVariableOpfeatures/bias^AssignVariableOp_33*
_output_shapes	
:?*
dtype0
?
Placeholder_34Placeholder*J
_output_shapes8
6:4????????????????????????????????????*
dtype0*?
shape6:4????????????????????????????????????
Q
AssignVariableOp_34AssignVariableOpprob/kernelPlaceholder_34*
dtype0
|
ReadVariableOp_34ReadVariableOpprob/kernel^AssignVariableOp_34*'
_output_shapes
:?*
dtype0
i
Placeholder_35Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
O
AssignVariableOp_35AssignVariableOp	prob/biasPlaceholder_35*
dtype0
m
ReadVariableOp_35ReadVariableOp	prob/bias^AssignVariableOp_35*
_output_shapes
:*
dtype0
?
Placeholder_36Placeholder*J
_output_shapes8
6:4????????????????????????????????????*
dtype0*?
shape6:4????????????????????????????????????
Q
AssignVariableOp_36AssignVariableOpdist/kernelPlaceholder_36*
dtype0
|
ReadVariableOp_36ReadVariableOpdist/kernel^AssignVariableOp_36*'
_output_shapes
:? *
dtype0
i
Placeholder_37Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
O
AssignVariableOp_37AssignVariableOp	dist/biasPlaceholder_37*
dtype0
m
ReadVariableOp_37ReadVariableOp	dist/bias^AssignVariableOp_37*
_output_shapes
: *
dtype0
?
Placeholder_38Placeholder*J
_output_shapes8
6:4????????????????????????????????????*
dtype0*?
shape6:4????????????????????????????????????
_
AssignVariableOp_38AssignVariableOpconv2d_transpose_1/kernelPlaceholder_38*
dtype0
?
ReadVariableOp_38ReadVariableOpconv2d_transpose_1/kernel^AssignVariableOp_38*&
_output_shapes
:*
dtype0
Z
VarIsInitializedOpVarIsInitializedOpdown_level_0_no_0/kernel*
_output_shapes
: 
\
VarIsInitializedOp_1VarIsInitializedOpdown_level_2_no_1/kernel*
_output_shapes
: 
Z
VarIsInitializedOp_2VarIsInitializedOpdown_level_2_no_1/bias*
_output_shapes
: 
X
VarIsInitializedOp_3VarIsInitializedOpup_level_1_no_0/bias*
_output_shapes
: 
Q
VarIsInitializedOp_4VarIsInitializedOpconv2d_5/bias*
_output_shapes
: 
Z
VarIsInitializedOp_5VarIsInitializedOpdown_level_2_no_0/bias*
_output_shapes
: 
\
VarIsInitializedOp_6VarIsInitializedOpdown_level_1_no_0/kernel*
_output_shapes
: 
Q
VarIsInitializedOp_7VarIsInitializedOpmiddle_2/bias*
_output_shapes
: 
Z
VarIsInitializedOp_8VarIsInitializedOpup_level_0_no_0/kernel*
_output_shapes
: 
X
VarIsInitializedOp_9VarIsInitializedOpup_level_0_no_0/bias*
_output_shapes
: 
[
VarIsInitializedOp_10VarIsInitializedOpdown_level_1_no_0/bias*
_output_shapes
: 
T
VarIsInitializedOp_11VarIsInitializedOpmiddle_0/kernel*
_output_shapes
: 
[
VarIsInitializedOp_12VarIsInitializedOpup_level_2_no_0/kernel*
_output_shapes
: 
Y
VarIsInitializedOp_13VarIsInitializedOpup_level_2_no_0/bias*
_output_shapes
: 
[
VarIsInitializedOp_14VarIsInitializedOpup_level_2_no_2/kernel*
_output_shapes
: 
[
VarIsInitializedOp_15VarIsInitializedOpup_level_1_no_2/kernel*
_output_shapes
: 
N
VarIsInitializedOp_16VarIsInitializedOp	dist/bias*
_output_shapes
: 
]
VarIsInitializedOp_17VarIsInitializedOpdown_level_1_no_1/kernel*
_output_shapes
: 
T
VarIsInitializedOp_18VarIsInitializedOpconv2d_4/kernel*
_output_shapes
: 
[
VarIsInitializedOp_19VarIsInitializedOpup_level_0_no_2/kernel*
_output_shapes
: 
^
VarIsInitializedOp_20VarIsInitializedOpconv2d_transpose_1/kernel*
_output_shapes
: 
[
VarIsInitializedOp_21VarIsInitializedOpdown_level_0_no_1/bias*
_output_shapes
: 
[
VarIsInitializedOp_22VarIsInitializedOpdown_level_1_no_1/bias*
_output_shapes
: 
Y
VarIsInitializedOp_23VarIsInitializedOpup_level_0_no_2/bias*
_output_shapes
: 
T
VarIsInitializedOp_24VarIsInitializedOpfeatures/kernel*
_output_shapes
: 
N
VarIsInitializedOp_25VarIsInitializedOp	prob/bias*
_output_shapes
: 
R
VarIsInitializedOp_26VarIsInitializedOpconv2d_4/bias*
_output_shapes
: 
T
VarIsInitializedOp_27VarIsInitializedOpconv2d_5/kernel*
_output_shapes
: 
R
VarIsInitializedOp_28VarIsInitializedOpmiddle_0/bias*
_output_shapes
: 
T
VarIsInitializedOp_29VarIsInitializedOpmiddle_2/kernel*
_output_shapes
: 
[
VarIsInitializedOp_30VarIsInitializedOpdown_level_0_no_0/bias*
_output_shapes
: 
]
VarIsInitializedOp_31VarIsInitializedOpdown_level_2_no_0/kernel*
_output_shapes
: 
Y
VarIsInitializedOp_32VarIsInitializedOpup_level_2_no_2/bias*
_output_shapes
: 
[
VarIsInitializedOp_33VarIsInitializedOpup_level_1_no_0/kernel*
_output_shapes
: 
Y
VarIsInitializedOp_34VarIsInitializedOpup_level_1_no_2/bias*
_output_shapes
: 
R
VarIsInitializedOp_35VarIsInitializedOpfeatures/bias*
_output_shapes
: 
P
VarIsInitializedOp_36VarIsInitializedOpprob/kernel*
_output_shapes
: 
P
VarIsInitializedOp_37VarIsInitializedOpdist/kernel*
_output_shapes
: 
]
VarIsInitializedOp_38VarIsInitializedOpdown_level_0_no_1/kernel*
_output_shapes
: 
?
initNoOp^conv2d_4/bias/Assign^conv2d_4/kernel/Assign^conv2d_5/bias/Assign^conv2d_5/kernel/Assign!^conv2d_transpose_1/kernel/Assign^dist/bias/Assign^dist/kernel/Assign^down_level_0_no_0/bias/Assign ^down_level_0_no_0/kernel/Assign^down_level_0_no_1/bias/Assign ^down_level_0_no_1/kernel/Assign^down_level_1_no_0/bias/Assign ^down_level_1_no_0/kernel/Assign^down_level_1_no_1/bias/Assign ^down_level_1_no_1/kernel/Assign^down_level_2_no_0/bias/Assign ^down_level_2_no_0/kernel/Assign^down_level_2_no_1/bias/Assign ^down_level_2_no_1/kernel/Assign^features/bias/Assign^features/kernel/Assign^middle_0/bias/Assign^middle_0/kernel/Assign^middle_2/bias/Assign^middle_2/kernel/Assign^prob/bias/Assign^prob/kernel/Assign^up_level_0_no_0/bias/Assign^up_level_0_no_0/kernel/Assign^up_level_0_no_2/bias/Assign^up_level_0_no_2/kernel/Assign^up_level_1_no_0/bias/Assign^up_level_1_no_0/kernel/Assign^up_level_1_no_2/bias/Assign^up_level_1_no_2/kernel/Assign^up_level_2_no_0/bias/Assign^up_level_2_no_0/kernel/Assign^up_level_2_no_2/bias/Assign^up_level_2_no_2/kernel/Assign
Y
save/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
n
save/filenamePlaceholderWithDefaultsave/filename/input*
_output_shapes
: *
dtype0*
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
_output_shapes
: *
dtype0*
shape: 
?
save/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_d3df42e59e4c476183d6633593dfe7ac/part
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
Q
save/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
k
save/ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
?
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
?
save/SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:'*
dtype0*?
value?B?'Bconv2d_4/biasBconv2d_4/kernelBconv2d_5/biasBconv2d_5/kernelBconv2d_transpose_1/kernelB	dist/biasBdist/kernelBdown_level_0_no_0/biasBdown_level_0_no_0/kernelBdown_level_0_no_1/biasBdown_level_0_no_1/kernelBdown_level_1_no_0/biasBdown_level_1_no_0/kernelBdown_level_1_no_1/biasBdown_level_1_no_1/kernelBdown_level_2_no_0/biasBdown_level_2_no_0/kernelBdown_level_2_no_1/biasBdown_level_2_no_1/kernelBfeatures/biasBfeatures/kernelBmiddle_0/biasBmiddle_0/kernelBmiddle_2/biasBmiddle_2/kernelB	prob/biasBprob/kernelBup_level_0_no_0/biasBup_level_0_no_0/kernelBup_level_0_no_2/biasBup_level_0_no_2/kernelBup_level_1_no_0/biasBup_level_1_no_0/kernelBup_level_1_no_2/biasBup_level_1_no_2/kernelBup_level_2_no_0/biasBup_level_2_no_0/kernelBup_level_2_no_2/biasBup_level_2_no_2/kernel
?
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:'*
dtype0*a
valueXBV'B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices!conv2d_4/bias/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_5/bias/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOp-conv2d_transpose_1/kernel/Read/ReadVariableOpdist/bias/Read/ReadVariableOpdist/kernel/Read/ReadVariableOp*down_level_0_no_0/bias/Read/ReadVariableOp,down_level_0_no_0/kernel/Read/ReadVariableOp*down_level_0_no_1/bias/Read/ReadVariableOp,down_level_0_no_1/kernel/Read/ReadVariableOp*down_level_1_no_0/bias/Read/ReadVariableOp,down_level_1_no_0/kernel/Read/ReadVariableOp*down_level_1_no_1/bias/Read/ReadVariableOp,down_level_1_no_1/kernel/Read/ReadVariableOp*down_level_2_no_0/bias/Read/ReadVariableOp,down_level_2_no_0/kernel/Read/ReadVariableOp*down_level_2_no_1/bias/Read/ReadVariableOp,down_level_2_no_1/kernel/Read/ReadVariableOp!features/bias/Read/ReadVariableOp#features/kernel/Read/ReadVariableOp!middle_0/bias/Read/ReadVariableOp#middle_0/kernel/Read/ReadVariableOp!middle_2/bias/Read/ReadVariableOp#middle_2/kernel/Read/ReadVariableOpprob/bias/Read/ReadVariableOpprob/kernel/Read/ReadVariableOp(up_level_0_no_0/bias/Read/ReadVariableOp*up_level_0_no_0/kernel/Read/ReadVariableOp(up_level_0_no_2/bias/Read/ReadVariableOp*up_level_0_no_2/kernel/Read/ReadVariableOp(up_level_1_no_0/bias/Read/ReadVariableOp*up_level_1_no_0/kernel/Read/ReadVariableOp(up_level_1_no_2/bias/Read/ReadVariableOp*up_level_1_no_2/kernel/Read/ReadVariableOp(up_level_2_no_0/bias/Read/ReadVariableOp*up_level_2_no_0/kernel/Read/ReadVariableOp(up_level_2_no_2/bias/Read/ReadVariableOp*up_level_2_no_2/kernel/Read/ReadVariableOp"/device:CPU:0*5
dtypes+
)2'
?
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
?
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
N*
T0*
_output_shapes
:*

axis 
?
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(
?
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
?
save/RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:'*
dtype0*?
value?B?'Bconv2d_4/biasBconv2d_4/kernelBconv2d_5/biasBconv2d_5/kernelBconv2d_transpose_1/kernelB	dist/biasBdist/kernelBdown_level_0_no_0/biasBdown_level_0_no_0/kernelBdown_level_0_no_1/biasBdown_level_0_no_1/kernelBdown_level_1_no_0/biasBdown_level_1_no_0/kernelBdown_level_1_no_1/biasBdown_level_1_no_1/kernelBdown_level_2_no_0/biasBdown_level_2_no_0/kernelBdown_level_2_no_1/biasBdown_level_2_no_1/kernelBfeatures/biasBfeatures/kernelBmiddle_0/biasBmiddle_0/kernelBmiddle_2/biasBmiddle_2/kernelB	prob/biasBprob/kernelBup_level_0_no_0/biasBup_level_0_no_0/kernelBup_level_0_no_2/biasBup_level_0_no_2/kernelBup_level_1_no_0/biasBup_level_1_no_0/kernelBup_level_1_no_2/biasBup_level_1_no_2/kernelBup_level_2_no_0/biasBup_level_2_no_0/kernelBup_level_2_no_2/biasBup_level_2_no_2/kernel
?
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:'*
dtype0*a
valueXBV'B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::*5
dtypes+
)2'
N
save/Identity_1Identitysave/RestoreV2*
T0*
_output_shapes
:
V
save/AssignVariableOpAssignVariableOpconv2d_4/biassave/Identity_1*
dtype0
P
save/Identity_2Identitysave/RestoreV2:1*
T0*
_output_shapes
:
Z
save/AssignVariableOp_1AssignVariableOpconv2d_4/kernelsave/Identity_2*
dtype0
P
save/Identity_3Identitysave/RestoreV2:2*
T0*
_output_shapes
:
X
save/AssignVariableOp_2AssignVariableOpconv2d_5/biassave/Identity_3*
dtype0
P
save/Identity_4Identitysave/RestoreV2:3*
T0*
_output_shapes
:
Z
save/AssignVariableOp_3AssignVariableOpconv2d_5/kernelsave/Identity_4*
dtype0
P
save/Identity_5Identitysave/RestoreV2:4*
T0*
_output_shapes
:
d
save/AssignVariableOp_4AssignVariableOpconv2d_transpose_1/kernelsave/Identity_5*
dtype0
P
save/Identity_6Identitysave/RestoreV2:5*
T0*
_output_shapes
:
T
save/AssignVariableOp_5AssignVariableOp	dist/biassave/Identity_6*
dtype0
P
save/Identity_7Identitysave/RestoreV2:6*
T0*
_output_shapes
:
V
save/AssignVariableOp_6AssignVariableOpdist/kernelsave/Identity_7*
dtype0
P
save/Identity_8Identitysave/RestoreV2:7*
T0*
_output_shapes
:
a
save/AssignVariableOp_7AssignVariableOpdown_level_0_no_0/biassave/Identity_8*
dtype0
P
save/Identity_9Identitysave/RestoreV2:8*
T0*
_output_shapes
:
c
save/AssignVariableOp_8AssignVariableOpdown_level_0_no_0/kernelsave/Identity_9*
dtype0
Q
save/Identity_10Identitysave/RestoreV2:9*
T0*
_output_shapes
:
b
save/AssignVariableOp_9AssignVariableOpdown_level_0_no_1/biassave/Identity_10*
dtype0
R
save/Identity_11Identitysave/RestoreV2:10*
T0*
_output_shapes
:
e
save/AssignVariableOp_10AssignVariableOpdown_level_0_no_1/kernelsave/Identity_11*
dtype0
R
save/Identity_12Identitysave/RestoreV2:11*
T0*
_output_shapes
:
c
save/AssignVariableOp_11AssignVariableOpdown_level_1_no_0/biassave/Identity_12*
dtype0
R
save/Identity_13Identitysave/RestoreV2:12*
T0*
_output_shapes
:
e
save/AssignVariableOp_12AssignVariableOpdown_level_1_no_0/kernelsave/Identity_13*
dtype0
R
save/Identity_14Identitysave/RestoreV2:13*
T0*
_output_shapes
:
c
save/AssignVariableOp_13AssignVariableOpdown_level_1_no_1/biassave/Identity_14*
dtype0
R
save/Identity_15Identitysave/RestoreV2:14*
T0*
_output_shapes
:
e
save/AssignVariableOp_14AssignVariableOpdown_level_1_no_1/kernelsave/Identity_15*
dtype0
R
save/Identity_16Identitysave/RestoreV2:15*
T0*
_output_shapes
:
c
save/AssignVariableOp_15AssignVariableOpdown_level_2_no_0/biassave/Identity_16*
dtype0
R
save/Identity_17Identitysave/RestoreV2:16*
T0*
_output_shapes
:
e
save/AssignVariableOp_16AssignVariableOpdown_level_2_no_0/kernelsave/Identity_17*
dtype0
R
save/Identity_18Identitysave/RestoreV2:17*
T0*
_output_shapes
:
c
save/AssignVariableOp_17AssignVariableOpdown_level_2_no_1/biassave/Identity_18*
dtype0
R
save/Identity_19Identitysave/RestoreV2:18*
T0*
_output_shapes
:
e
save/AssignVariableOp_18AssignVariableOpdown_level_2_no_1/kernelsave/Identity_19*
dtype0
R
save/Identity_20Identitysave/RestoreV2:19*
T0*
_output_shapes
:
Z
save/AssignVariableOp_19AssignVariableOpfeatures/biassave/Identity_20*
dtype0
R
save/Identity_21Identitysave/RestoreV2:20*
T0*
_output_shapes
:
\
save/AssignVariableOp_20AssignVariableOpfeatures/kernelsave/Identity_21*
dtype0
R
save/Identity_22Identitysave/RestoreV2:21*
T0*
_output_shapes
:
Z
save/AssignVariableOp_21AssignVariableOpmiddle_0/biassave/Identity_22*
dtype0
R
save/Identity_23Identitysave/RestoreV2:22*
T0*
_output_shapes
:
\
save/AssignVariableOp_22AssignVariableOpmiddle_0/kernelsave/Identity_23*
dtype0
R
save/Identity_24Identitysave/RestoreV2:23*
T0*
_output_shapes
:
Z
save/AssignVariableOp_23AssignVariableOpmiddle_2/biassave/Identity_24*
dtype0
R
save/Identity_25Identitysave/RestoreV2:24*
T0*
_output_shapes
:
\
save/AssignVariableOp_24AssignVariableOpmiddle_2/kernelsave/Identity_25*
dtype0
R
save/Identity_26Identitysave/RestoreV2:25*
T0*
_output_shapes
:
V
save/AssignVariableOp_25AssignVariableOp	prob/biassave/Identity_26*
dtype0
R
save/Identity_27Identitysave/RestoreV2:26*
T0*
_output_shapes
:
X
save/AssignVariableOp_26AssignVariableOpprob/kernelsave/Identity_27*
dtype0
R
save/Identity_28Identitysave/RestoreV2:27*
T0*
_output_shapes
:
a
save/AssignVariableOp_27AssignVariableOpup_level_0_no_0/biassave/Identity_28*
dtype0
R
save/Identity_29Identitysave/RestoreV2:28*
T0*
_output_shapes
:
c
save/AssignVariableOp_28AssignVariableOpup_level_0_no_0/kernelsave/Identity_29*
dtype0
R
save/Identity_30Identitysave/RestoreV2:29*
T0*
_output_shapes
:
a
save/AssignVariableOp_29AssignVariableOpup_level_0_no_2/biassave/Identity_30*
dtype0
R
save/Identity_31Identitysave/RestoreV2:30*
T0*
_output_shapes
:
c
save/AssignVariableOp_30AssignVariableOpup_level_0_no_2/kernelsave/Identity_31*
dtype0
R
save/Identity_32Identitysave/RestoreV2:31*
T0*
_output_shapes
:
a
save/AssignVariableOp_31AssignVariableOpup_level_1_no_0/biassave/Identity_32*
dtype0
R
save/Identity_33Identitysave/RestoreV2:32*
T0*
_output_shapes
:
c
save/AssignVariableOp_32AssignVariableOpup_level_1_no_0/kernelsave/Identity_33*
dtype0
R
save/Identity_34Identitysave/RestoreV2:33*
T0*
_output_shapes
:
a
save/AssignVariableOp_33AssignVariableOpup_level_1_no_2/biassave/Identity_34*
dtype0
R
save/Identity_35Identitysave/RestoreV2:34*
T0*
_output_shapes
:
c
save/AssignVariableOp_34AssignVariableOpup_level_1_no_2/kernelsave/Identity_35*
dtype0
R
save/Identity_36Identitysave/RestoreV2:35*
T0*
_output_shapes
:
a
save/AssignVariableOp_35AssignVariableOpup_level_2_no_0/biassave/Identity_36*
dtype0
R
save/Identity_37Identitysave/RestoreV2:36*
T0*
_output_shapes
:
c
save/AssignVariableOp_36AssignVariableOpup_level_2_no_0/kernelsave/Identity_37*
dtype0
R
save/Identity_38Identitysave/RestoreV2:37*
T0*
_output_shapes
:
a
save/AssignVariableOp_37AssignVariableOpup_level_2_no_2/biassave/Identity_38*
dtype0
R
save/Identity_39Identitysave/RestoreV2:38*
T0*
_output_shapes
:
c
save/AssignVariableOp_38AssignVariableOpup_level_2_no_2/kernelsave/Identity_39*
dtype0
?
save/restore_shardNoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_10^save/AssignVariableOp_11^save/AssignVariableOp_12^save/AssignVariableOp_13^save/AssignVariableOp_14^save/AssignVariableOp_15^save/AssignVariableOp_16^save/AssignVariableOp_17^save/AssignVariableOp_18^save/AssignVariableOp_19^save/AssignVariableOp_2^save/AssignVariableOp_20^save/AssignVariableOp_21^save/AssignVariableOp_22^save/AssignVariableOp_23^save/AssignVariableOp_24^save/AssignVariableOp_25^save/AssignVariableOp_26^save/AssignVariableOp_27^save/AssignVariableOp_28^save/AssignVariableOp_29^save/AssignVariableOp_3^save/AssignVariableOp_30^save/AssignVariableOp_31^save/AssignVariableOp_32^save/AssignVariableOp_33^save/AssignVariableOp_34^save/AssignVariableOp_35^save/AssignVariableOp_36^save/AssignVariableOp_37^save/AssignVariableOp_38^save/AssignVariableOp_4^save/AssignVariableOp_5^save/AssignVariableOp_6^save/AssignVariableOp_7^save/AssignVariableOp_8^save/AssignVariableOp_9
-
save/restore_allNoOp^save/restore_shard"?<
save/Const:0save/Identity:0save/restore_all (5 @F8"?,
trainable_variables?,?,
?
conv2d_4/kernel:0conv2d_4/kernel/Assign%conv2d_4/kernel/Read/ReadVariableOp:0(2,conv2d_4/kernel/Initializer/random_uniform:08
s
conv2d_4/bias:0conv2d_4/bias/Assign#conv2d_4/bias/Read/ReadVariableOp:0(2!conv2d_4/bias/Initializer/zeros:08
?
conv2d_5/kernel:0conv2d_5/kernel/Assign%conv2d_5/kernel/Read/ReadVariableOp:0(2,conv2d_5/kernel/Initializer/random_uniform:08
s
conv2d_5/bias:0conv2d_5/bias/Assign#conv2d_5/bias/Read/ReadVariableOp:0(2!conv2d_5/bias/Initializer/zeros:08
?
down_level_0_no_0/kernel:0down_level_0_no_0/kernel/Assign.down_level_0_no_0/kernel/Read/ReadVariableOp:0(25down_level_0_no_0/kernel/Initializer/random_uniform:08
?
down_level_0_no_0/bias:0down_level_0_no_0/bias/Assign,down_level_0_no_0/bias/Read/ReadVariableOp:0(2*down_level_0_no_0/bias/Initializer/zeros:08
?
down_level_0_no_1/kernel:0down_level_0_no_1/kernel/Assign.down_level_0_no_1/kernel/Read/ReadVariableOp:0(25down_level_0_no_1/kernel/Initializer/random_uniform:08
?
down_level_0_no_1/bias:0down_level_0_no_1/bias/Assign,down_level_0_no_1/bias/Read/ReadVariableOp:0(2*down_level_0_no_1/bias/Initializer/zeros:08
?
down_level_1_no_0/kernel:0down_level_1_no_0/kernel/Assign.down_level_1_no_0/kernel/Read/ReadVariableOp:0(25down_level_1_no_0/kernel/Initializer/random_uniform:08
?
down_level_1_no_0/bias:0down_level_1_no_0/bias/Assign,down_level_1_no_0/bias/Read/ReadVariableOp:0(2*down_level_1_no_0/bias/Initializer/zeros:08
?
down_level_1_no_1/kernel:0down_level_1_no_1/kernel/Assign.down_level_1_no_1/kernel/Read/ReadVariableOp:0(25down_level_1_no_1/kernel/Initializer/random_uniform:08
?
down_level_1_no_1/bias:0down_level_1_no_1/bias/Assign,down_level_1_no_1/bias/Read/ReadVariableOp:0(2*down_level_1_no_1/bias/Initializer/zeros:08
?
down_level_2_no_0/kernel:0down_level_2_no_0/kernel/Assign.down_level_2_no_0/kernel/Read/ReadVariableOp:0(25down_level_2_no_0/kernel/Initializer/random_uniform:08
?
down_level_2_no_0/bias:0down_level_2_no_0/bias/Assign,down_level_2_no_0/bias/Read/ReadVariableOp:0(2*down_level_2_no_0/bias/Initializer/zeros:08
?
down_level_2_no_1/kernel:0down_level_2_no_1/kernel/Assign.down_level_2_no_1/kernel/Read/ReadVariableOp:0(25down_level_2_no_1/kernel/Initializer/random_uniform:08
?
down_level_2_no_1/bias:0down_level_2_no_1/bias/Assign,down_level_2_no_1/bias/Read/ReadVariableOp:0(2*down_level_2_no_1/bias/Initializer/zeros:08
?
middle_0/kernel:0middle_0/kernel/Assign%middle_0/kernel/Read/ReadVariableOp:0(2,middle_0/kernel/Initializer/random_uniform:08
s
middle_0/bias:0middle_0/bias/Assign#middle_0/bias/Read/ReadVariableOp:0(2!middle_0/bias/Initializer/zeros:08
?
middle_2/kernel:0middle_2/kernel/Assign%middle_2/kernel/Read/ReadVariableOp:0(2,middle_2/kernel/Initializer/random_uniform:08
s
middle_2/bias:0middle_2/bias/Assign#middle_2/bias/Read/ReadVariableOp:0(2!middle_2/bias/Initializer/zeros:08
?
up_level_2_no_0/kernel:0up_level_2_no_0/kernel/Assign,up_level_2_no_0/kernel/Read/ReadVariableOp:0(23up_level_2_no_0/kernel/Initializer/random_uniform:08
?
up_level_2_no_0/bias:0up_level_2_no_0/bias/Assign*up_level_2_no_0/bias/Read/ReadVariableOp:0(2(up_level_2_no_0/bias/Initializer/zeros:08
?
up_level_2_no_2/kernel:0up_level_2_no_2/kernel/Assign,up_level_2_no_2/kernel/Read/ReadVariableOp:0(23up_level_2_no_2/kernel/Initializer/random_uniform:08
?
up_level_2_no_2/bias:0up_level_2_no_2/bias/Assign*up_level_2_no_2/bias/Read/ReadVariableOp:0(2(up_level_2_no_2/bias/Initializer/zeros:08
?
up_level_1_no_0/kernel:0up_level_1_no_0/kernel/Assign,up_level_1_no_0/kernel/Read/ReadVariableOp:0(23up_level_1_no_0/kernel/Initializer/random_uniform:08
?
up_level_1_no_0/bias:0up_level_1_no_0/bias/Assign*up_level_1_no_0/bias/Read/ReadVariableOp:0(2(up_level_1_no_0/bias/Initializer/zeros:08
?
up_level_1_no_2/kernel:0up_level_1_no_2/kernel/Assign,up_level_1_no_2/kernel/Read/ReadVariableOp:0(23up_level_1_no_2/kernel/Initializer/random_uniform:08
?
up_level_1_no_2/bias:0up_level_1_no_2/bias/Assign*up_level_1_no_2/bias/Read/ReadVariableOp:0(2(up_level_1_no_2/bias/Initializer/zeros:08
?
up_level_0_no_0/kernel:0up_level_0_no_0/kernel/Assign,up_level_0_no_0/kernel/Read/ReadVariableOp:0(23up_level_0_no_0/kernel/Initializer/random_uniform:08
?
up_level_0_no_0/bias:0up_level_0_no_0/bias/Assign*up_level_0_no_0/bias/Read/ReadVariableOp:0(2(up_level_0_no_0/bias/Initializer/zeros:08
?
up_level_0_no_2/kernel:0up_level_0_no_2/kernel/Assign,up_level_0_no_2/kernel/Read/ReadVariableOp:0(23up_level_0_no_2/kernel/Initializer/random_uniform:08
?
up_level_0_no_2/bias:0up_level_0_no_2/bias/Assign*up_level_0_no_2/bias/Read/ReadVariableOp:0(2(up_level_0_no_2/bias/Initializer/zeros:08
?
features/kernel:0features/kernel/Assign%features/kernel/Read/ReadVariableOp:0(2,features/kernel/Initializer/random_uniform:08
s
features/bias:0features/bias/Assign#features/bias/Read/ReadVariableOp:0(2!features/bias/Initializer/zeros:08
t
prob/kernel:0prob/kernel/Assign!prob/kernel/Read/ReadVariableOp:0(2(prob/kernel/Initializer/random_uniform:08
c
prob/bias:0prob/bias/Assignprob/bias/Read/ReadVariableOp:0(2prob/bias/Initializer/zeros:08
t
dist/kernel:0dist/kernel/Assign!dist/kernel/Read/ReadVariableOp:0(2(dist/kernel/Initializer/random_uniform:08
c
dist/bias:0dist/bias/Assigndist/bias/Read/ReadVariableOp:0(2dist/bias/Initializer/zeros:08
?
conv2d_transpose_1/kernel:0 conv2d_transpose_1/kernel/Assign/conv2d_transpose_1/kernel/Read/ReadVariableOp:0(2,conv2d_transpose_1/kernel/Initializer/ones:08"?,
	variables?,?,
?
conv2d_4/kernel:0conv2d_4/kernel/Assign%conv2d_4/kernel/Read/ReadVariableOp:0(2,conv2d_4/kernel/Initializer/random_uniform:08
s
conv2d_4/bias:0conv2d_4/bias/Assign#conv2d_4/bias/Read/ReadVariableOp:0(2!conv2d_4/bias/Initializer/zeros:08
?
conv2d_5/kernel:0conv2d_5/kernel/Assign%conv2d_5/kernel/Read/ReadVariableOp:0(2,conv2d_5/kernel/Initializer/random_uniform:08
s
conv2d_5/bias:0conv2d_5/bias/Assign#conv2d_5/bias/Read/ReadVariableOp:0(2!conv2d_5/bias/Initializer/zeros:08
?
down_level_0_no_0/kernel:0down_level_0_no_0/kernel/Assign.down_level_0_no_0/kernel/Read/ReadVariableOp:0(25down_level_0_no_0/kernel/Initializer/random_uniform:08
?
down_level_0_no_0/bias:0down_level_0_no_0/bias/Assign,down_level_0_no_0/bias/Read/ReadVariableOp:0(2*down_level_0_no_0/bias/Initializer/zeros:08
?
down_level_0_no_1/kernel:0down_level_0_no_1/kernel/Assign.down_level_0_no_1/kernel/Read/ReadVariableOp:0(25down_level_0_no_1/kernel/Initializer/random_uniform:08
?
down_level_0_no_1/bias:0down_level_0_no_1/bias/Assign,down_level_0_no_1/bias/Read/ReadVariableOp:0(2*down_level_0_no_1/bias/Initializer/zeros:08
?
down_level_1_no_0/kernel:0down_level_1_no_0/kernel/Assign.down_level_1_no_0/kernel/Read/ReadVariableOp:0(25down_level_1_no_0/kernel/Initializer/random_uniform:08
?
down_level_1_no_0/bias:0down_level_1_no_0/bias/Assign,down_level_1_no_0/bias/Read/ReadVariableOp:0(2*down_level_1_no_0/bias/Initializer/zeros:08
?
down_level_1_no_1/kernel:0down_level_1_no_1/kernel/Assign.down_level_1_no_1/kernel/Read/ReadVariableOp:0(25down_level_1_no_1/kernel/Initializer/random_uniform:08
?
down_level_1_no_1/bias:0down_level_1_no_1/bias/Assign,down_level_1_no_1/bias/Read/ReadVariableOp:0(2*down_level_1_no_1/bias/Initializer/zeros:08
?
down_level_2_no_0/kernel:0down_level_2_no_0/kernel/Assign.down_level_2_no_0/kernel/Read/ReadVariableOp:0(25down_level_2_no_0/kernel/Initializer/random_uniform:08
?
down_level_2_no_0/bias:0down_level_2_no_0/bias/Assign,down_level_2_no_0/bias/Read/ReadVariableOp:0(2*down_level_2_no_0/bias/Initializer/zeros:08
?
down_level_2_no_1/kernel:0down_level_2_no_1/kernel/Assign.down_level_2_no_1/kernel/Read/ReadVariableOp:0(25down_level_2_no_1/kernel/Initializer/random_uniform:08
?
down_level_2_no_1/bias:0down_level_2_no_1/bias/Assign,down_level_2_no_1/bias/Read/ReadVariableOp:0(2*down_level_2_no_1/bias/Initializer/zeros:08
?
middle_0/kernel:0middle_0/kernel/Assign%middle_0/kernel/Read/ReadVariableOp:0(2,middle_0/kernel/Initializer/random_uniform:08
s
middle_0/bias:0middle_0/bias/Assign#middle_0/bias/Read/ReadVariableOp:0(2!middle_0/bias/Initializer/zeros:08
?
middle_2/kernel:0middle_2/kernel/Assign%middle_2/kernel/Read/ReadVariableOp:0(2,middle_2/kernel/Initializer/random_uniform:08
s
middle_2/bias:0middle_2/bias/Assign#middle_2/bias/Read/ReadVariableOp:0(2!middle_2/bias/Initializer/zeros:08
?
up_level_2_no_0/kernel:0up_level_2_no_0/kernel/Assign,up_level_2_no_0/kernel/Read/ReadVariableOp:0(23up_level_2_no_0/kernel/Initializer/random_uniform:08
?
up_level_2_no_0/bias:0up_level_2_no_0/bias/Assign*up_level_2_no_0/bias/Read/ReadVariableOp:0(2(up_level_2_no_0/bias/Initializer/zeros:08
?
up_level_2_no_2/kernel:0up_level_2_no_2/kernel/Assign,up_level_2_no_2/kernel/Read/ReadVariableOp:0(23up_level_2_no_2/kernel/Initializer/random_uniform:08
?
up_level_2_no_2/bias:0up_level_2_no_2/bias/Assign*up_level_2_no_2/bias/Read/ReadVariableOp:0(2(up_level_2_no_2/bias/Initializer/zeros:08
?
up_level_1_no_0/kernel:0up_level_1_no_0/kernel/Assign,up_level_1_no_0/kernel/Read/ReadVariableOp:0(23up_level_1_no_0/kernel/Initializer/random_uniform:08
?
up_level_1_no_0/bias:0up_level_1_no_0/bias/Assign*up_level_1_no_0/bias/Read/ReadVariableOp:0(2(up_level_1_no_0/bias/Initializer/zeros:08
?
up_level_1_no_2/kernel:0up_level_1_no_2/kernel/Assign,up_level_1_no_2/kernel/Read/ReadVariableOp:0(23up_level_1_no_2/kernel/Initializer/random_uniform:08
?
up_level_1_no_2/bias:0up_level_1_no_2/bias/Assign*up_level_1_no_2/bias/Read/ReadVariableOp:0(2(up_level_1_no_2/bias/Initializer/zeros:08
?
up_level_0_no_0/kernel:0up_level_0_no_0/kernel/Assign,up_level_0_no_0/kernel/Read/ReadVariableOp:0(23up_level_0_no_0/kernel/Initializer/random_uniform:08
?
up_level_0_no_0/bias:0up_level_0_no_0/bias/Assign*up_level_0_no_0/bias/Read/ReadVariableOp:0(2(up_level_0_no_0/bias/Initializer/zeros:08
?
up_level_0_no_2/kernel:0up_level_0_no_2/kernel/Assign,up_level_0_no_2/kernel/Read/ReadVariableOp:0(23up_level_0_no_2/kernel/Initializer/random_uniform:08
?
up_level_0_no_2/bias:0up_level_0_no_2/bias/Assign*up_level_0_no_2/bias/Read/ReadVariableOp:0(2(up_level_0_no_2/bias/Initializer/zeros:08
?
features/kernel:0features/kernel/Assign%features/kernel/Read/ReadVariableOp:0(2,features/kernel/Initializer/random_uniform:08
s
features/bias:0features/bias/Assign#features/bias/Read/ReadVariableOp:0(2!features/bias/Initializer/zeros:08
t
prob/kernel:0prob/kernel/Assign!prob/kernel/Read/ReadVariableOp:0(2(prob/kernel/Initializer/random_uniform:08
c
prob/bias:0prob/bias/Assignprob/bias/Read/ReadVariableOp:0(2prob/bias/Initializer/zeros:08
t
dist/kernel:0dist/kernel/Assign!dist/kernel/Read/ReadVariableOp:0(2(dist/kernel/Initializer/random_uniform:08
c
dist/bias:0dist/bias/Assigndist/bias/Read/ReadVariableOp:0(2dist/bias/Initializer/zeros:08
?
conv2d_transpose_1/kernel:0 conv2d_transpose_1/kernel/Assign/conv2d_transpose_1/kernel/Read/ReadVariableOp:0(2,conv2d_transpose_1/kernel/Initializer/ones:08*?
serving_default?
A
input8
input:0+???????????????????????????R
outputH
concatenate_10/concat:0+???????????????????????????!tensorflow/serving/predict