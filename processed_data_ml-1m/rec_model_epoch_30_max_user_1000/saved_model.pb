??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.1.02v2.1.0-rc2-17-ge5bf8de8??
?
mlp_embedding_user/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?7 *.
shared_namemlp_embedding_user/embeddings
?
1mlp_embedding_user/embeddings/Read/ReadVariableOpReadVariableOpmlp_embedding_user/embeddings*
_output_shapes
:	?7 *
dtype0
?
mlp_embedding_item/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *.
shared_namemlp_embedding_item/embeddings
?
1mlp_embedding_item/embeddings/Read/ReadVariableOpReadVariableOpmlp_embedding_item/embeddings*
_output_shapes
:	? *
dtype0
?
mf_embedding_user/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?7*-
shared_namemf_embedding_user/embeddings
?
0mf_embedding_user/embeddings/Read/ReadVariableOpReadVariableOpmf_embedding_user/embeddings*
_output_shapes
:	?7*
dtype0
?
mf_embedding_item/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*-
shared_namemf_embedding_item/embeddings
?
0mf_embedding_item/embeddings/Read/ReadVariableOpReadVariableOpmf_embedding_item/embeddings*
_output_shapes
:	?*
dtype0
v
layer1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *
shared_namelayer1/kernel
o
!layer1/kernel/Read/ReadVariableOpReadVariableOplayer1/kernel*
_output_shapes

:@ *
dtype0
n
layer1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelayer1/bias
g
layer1/bias/Read/ReadVariableOpReadVariableOplayer1/bias*
_output_shapes
: *
dtype0
v
layer2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namelayer2/kernel
o
!layer2/kernel/Read/ReadVariableOpReadVariableOplayer2/kernel*
_output_shapes

: *
dtype0
n
layer2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namelayer2/bias
g
layer2/bias/Read/ReadVariableOpReadVariableOplayer2/bias*
_output_shapes
:*
dtype0
v
layer3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namelayer3/kernel
o
!layer3/kernel/Read/ReadVariableOpReadVariableOplayer3/kernel*
_output_shapes

:*
dtype0
n
layer3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namelayer3/bias
g
layer3/bias/Read/ReadVariableOpReadVariableOplayer3/bias*
_output_shapes
:*
dtype0
~
prediction/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_nameprediction/kernel
w
%prediction/kernel/Read/ReadVariableOpReadVariableOpprediction/kernel*
_output_shapes

:*
dtype0
v
prediction/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameprediction/bias
o
#prediction/bias/Read/ReadVariableOpReadVariableOpprediction/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
$Adam/mlp_embedding_user/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?7 *5
shared_name&$Adam/mlp_embedding_user/embeddings/m
?
8Adam/mlp_embedding_user/embeddings/m/Read/ReadVariableOpReadVariableOp$Adam/mlp_embedding_user/embeddings/m*
_output_shapes
:	?7 *
dtype0
?
$Adam/mlp_embedding_item/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *5
shared_name&$Adam/mlp_embedding_item/embeddings/m
?
8Adam/mlp_embedding_item/embeddings/m/Read/ReadVariableOpReadVariableOp$Adam/mlp_embedding_item/embeddings/m*
_output_shapes
:	? *
dtype0
?
#Adam/mf_embedding_user/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?7*4
shared_name%#Adam/mf_embedding_user/embeddings/m
?
7Adam/mf_embedding_user/embeddings/m/Read/ReadVariableOpReadVariableOp#Adam/mf_embedding_user/embeddings/m*
_output_shapes
:	?7*
dtype0
?
#Adam/mf_embedding_item/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*4
shared_name%#Adam/mf_embedding_item/embeddings/m
?
7Adam/mf_embedding_item/embeddings/m/Read/ReadVariableOpReadVariableOp#Adam/mf_embedding_item/embeddings/m*
_output_shapes
:	?*
dtype0
?
Adam/layer1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *%
shared_nameAdam/layer1/kernel/m
}
(Adam/layer1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer1/kernel/m*
_output_shapes

:@ *
dtype0
|
Adam/layer1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/layer1/bias/m
u
&Adam/layer1/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer1/bias/m*
_output_shapes
: *
dtype0
?
Adam/layer2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *%
shared_nameAdam/layer2/kernel/m
}
(Adam/layer2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer2/kernel/m*
_output_shapes

: *
dtype0
|
Adam/layer2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/layer2/bias/m
u
&Adam/layer2/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer2/bias/m*
_output_shapes
:*
dtype0
?
Adam/layer3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameAdam/layer3/kernel/m
}
(Adam/layer3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer3/kernel/m*
_output_shapes

:*
dtype0
|
Adam/layer3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/layer3/bias/m
u
&Adam/layer3/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer3/bias/m*
_output_shapes
:*
dtype0
?
Adam/prediction/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/prediction/kernel/m
?
,Adam/prediction/kernel/m/Read/ReadVariableOpReadVariableOpAdam/prediction/kernel/m*
_output_shapes

:*
dtype0
?
Adam/prediction/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/prediction/bias/m
}
*Adam/prediction/bias/m/Read/ReadVariableOpReadVariableOpAdam/prediction/bias/m*
_output_shapes
:*
dtype0
?
$Adam/mlp_embedding_user/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?7 *5
shared_name&$Adam/mlp_embedding_user/embeddings/v
?
8Adam/mlp_embedding_user/embeddings/v/Read/ReadVariableOpReadVariableOp$Adam/mlp_embedding_user/embeddings/v*
_output_shapes
:	?7 *
dtype0
?
$Adam/mlp_embedding_item/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *5
shared_name&$Adam/mlp_embedding_item/embeddings/v
?
8Adam/mlp_embedding_item/embeddings/v/Read/ReadVariableOpReadVariableOp$Adam/mlp_embedding_item/embeddings/v*
_output_shapes
:	? *
dtype0
?
#Adam/mf_embedding_user/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?7*4
shared_name%#Adam/mf_embedding_user/embeddings/v
?
7Adam/mf_embedding_user/embeddings/v/Read/ReadVariableOpReadVariableOp#Adam/mf_embedding_user/embeddings/v*
_output_shapes
:	?7*
dtype0
?
#Adam/mf_embedding_item/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*4
shared_name%#Adam/mf_embedding_item/embeddings/v
?
7Adam/mf_embedding_item/embeddings/v/Read/ReadVariableOpReadVariableOp#Adam/mf_embedding_item/embeddings/v*
_output_shapes
:	?*
dtype0
?
Adam/layer1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *%
shared_nameAdam/layer1/kernel/v
}
(Adam/layer1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer1/kernel/v*
_output_shapes

:@ *
dtype0
|
Adam/layer1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/layer1/bias/v
u
&Adam/layer1/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer1/bias/v*
_output_shapes
: *
dtype0
?
Adam/layer2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *%
shared_nameAdam/layer2/kernel/v
}
(Adam/layer2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer2/kernel/v*
_output_shapes

: *
dtype0
|
Adam/layer2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/layer2/bias/v
u
&Adam/layer2/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer2/bias/v*
_output_shapes
:*
dtype0
?
Adam/layer3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameAdam/layer3/kernel/v
}
(Adam/layer3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer3/kernel/v*
_output_shapes

:*
dtype0
|
Adam/layer3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/layer3/bias/v
u
&Adam/layer3/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer3/bias/v*
_output_shapes
:*
dtype0
?
Adam/prediction/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/prediction/kernel/v
?
,Adam/prediction/kernel/v/Read/ReadVariableOpReadVariableOpAdam/prediction/kernel/v*
_output_shapes

:*
dtype0
?
Adam/prediction/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/prediction/bias/v
}
*Adam/prediction/bias/v/Read/ReadVariableOpReadVariableOpAdam/prediction/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?T
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?T
value?TB?T B?T
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer_with_weights-3
	layer-8

layer_with_weights-4

layer-9
layer-10
layer-11
layer_with_weights-5
layer-12
layer-13
layer_with_weights-6
layer-14
layer-15
layer_with_weights-7
layer-16
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
 
 
b

embeddings
regularization_losses
	variables
trainable_variables
	keras_api
b

embeddings
regularization_losses
	variables
 trainable_variables
!	keras_api
R
"regularization_losses
#	variables
$trainable_variables
%	keras_api
R
&regularization_losses
'	variables
(trainable_variables
)	keras_api
R
*regularization_losses
+	variables
,trainable_variables
-	keras_api
b
.
embeddings
/regularization_losses
0	variables
1trainable_variables
2	keras_api
b
3
embeddings
4regularization_losses
5	variables
6trainable_variables
7	keras_api
h

8kernel
9bias
:regularization_losses
;	variables
<trainable_variables
=	keras_api
R
>regularization_losses
?	variables
@trainable_variables
A	keras_api
R
Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api
h

Fkernel
Gbias
Hregularization_losses
I	variables
Jtrainable_variables
K	keras_api
R
Lregularization_losses
M	variables
Ntrainable_variables
O	keras_api
h

Pkernel
Qbias
Rregularization_losses
S	variables
Ttrainable_variables
U	keras_api
R
Vregularization_losses
W	variables
Xtrainable_variables
Y	keras_api
h

Zkernel
[bias
\regularization_losses
]	variables
^trainable_variables
_	keras_api
?
`iter

abeta_1

bbeta_2
	cdecay
dlearning_ratem?m?.m?3m?8m?9m?Fm?Gm?Pm?Qm?Zm?[m?v?v?.v?3v?8v?9v?Fv?Gv?Pv?Qv?Zv?[v?
 
V
0
1
.2
33
84
95
F6
G7
P8
Q9
Z10
[11
V
0
1
.2
33
84
95
F6
G7
P8
Q9
Z10
[11
?
emetrics
regularization_losses

flayers
	variables
glayer_regularization_losses
hnon_trainable_variables
trainable_variables
 
mk
VARIABLE_VALUEmlp_embedding_user/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
?
imetrics
regularization_losses

jlayers
	variables
klayer_regularization_losses
lnon_trainable_variables
trainable_variables
mk
VARIABLE_VALUEmlp_embedding_item/embeddings:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
?
mmetrics
regularization_losses

nlayers
	variables
olayer_regularization_losses
pnon_trainable_variables
 trainable_variables
 
 
 
?
qmetrics
"regularization_losses

rlayers
#	variables
slayer_regularization_losses
tnon_trainable_variables
$trainable_variables
 
 
 
?
umetrics
&regularization_losses

vlayers
'	variables
wlayer_regularization_losses
xnon_trainable_variables
(trainable_variables
 
 
 
?
ymetrics
*regularization_losses

zlayers
+	variables
{layer_regularization_losses
|non_trainable_variables
,trainable_variables
lj
VARIABLE_VALUEmf_embedding_user/embeddings:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUE
 

.0

.0
?
}metrics
/regularization_losses

~layers
0	variables
layer_regularization_losses
?non_trainable_variables
1trainable_variables
lj
VARIABLE_VALUEmf_embedding_item/embeddings:layer_with_weights-3/embeddings/.ATTRIBUTES/VARIABLE_VALUE
 

30

30
?
?metrics
4regularization_losses
?layers
5	variables
 ?layer_regularization_losses
?non_trainable_variables
6trainable_variables
YW
VARIABLE_VALUElayer1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

80
91

80
91
?
?metrics
:regularization_losses
?layers
;	variables
 ?layer_regularization_losses
?non_trainable_variables
<trainable_variables
 
 
 
?
?metrics
>regularization_losses
?layers
?	variables
 ?layer_regularization_losses
?non_trainable_variables
@trainable_variables
 
 
 
?
?metrics
Bregularization_losses
?layers
C	variables
 ?layer_regularization_losses
?non_trainable_variables
Dtrainable_variables
YW
VARIABLE_VALUElayer2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

F0
G1

F0
G1
?
?metrics
Hregularization_losses
?layers
I	variables
 ?layer_regularization_losses
?non_trainable_variables
Jtrainable_variables
 
 
 
?
?metrics
Lregularization_losses
?layers
M	variables
 ?layer_regularization_losses
?non_trainable_variables
Ntrainable_variables
YW
VARIABLE_VALUElayer3/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer3/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

P0
Q1

P0
Q1
?
?metrics
Rregularization_losses
?layers
S	variables
 ?layer_regularization_losses
?non_trainable_variables
Ttrainable_variables
 
 
 
?
?metrics
Vregularization_losses
?layers
W	variables
 ?layer_regularization_losses
?non_trainable_variables
Xtrainable_variables
][
VARIABLE_VALUEprediction/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEprediction/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

Z0
[1

Z0
[1
?
?metrics
\regularization_losses
?layers
]	variables
 ?layer_regularization_losses
?non_trainable_variables
^trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

?0
~
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 


?total

?count
?
_fn_kwargs
?regularization_losses
?	variables
?trainable_variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 

?0
?1
 
?
?metrics
?regularization_losses
?layers
?	variables
 ?layer_regularization_losses
?non_trainable_variables
?trainable_variables
 
 
 

?0
?1
??
VARIABLE_VALUE$Adam/mlp_embedding_user/embeddings/mVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/mlp_embedding_item/embeddings/mVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/mf_embedding_user/embeddings/mVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/mf_embedding_item/embeddings/mVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer1/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer1/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer2/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer2/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer3/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer3/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/prediction/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/prediction/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/mlp_embedding_user/embeddings/vVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/mlp_embedding_item/embeddings/vVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/mf_embedding_user/embeddings/vVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/mf_embedding_item/embeddings/vVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer1/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer1/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer2/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer2/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer3/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer3/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/prediction/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/prediction/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
serving_default_item_inputPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
}
serving_default_user_inputPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_item_inputserving_default_user_inputmlp_embedding_item/embeddingsmlp_embedding_user/embeddingslayer1/kernellayer1/biasmf_embedding_item/embeddingsmf_embedding_user/embeddingslayer2/kernellayer2/biaslayer3/kernellayer3/biasprediction/kernelprediction/bias*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*-
f(R&
$__inference_signature_wrapper_876312
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename1mlp_embedding_user/embeddings/Read/ReadVariableOp1mlp_embedding_item/embeddings/Read/ReadVariableOp0mf_embedding_user/embeddings/Read/ReadVariableOp0mf_embedding_item/embeddings/Read/ReadVariableOp!layer1/kernel/Read/ReadVariableOplayer1/bias/Read/ReadVariableOp!layer2/kernel/Read/ReadVariableOplayer2/bias/Read/ReadVariableOp!layer3/kernel/Read/ReadVariableOplayer3/bias/Read/ReadVariableOp%prediction/kernel/Read/ReadVariableOp#prediction/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp8Adam/mlp_embedding_user/embeddings/m/Read/ReadVariableOp8Adam/mlp_embedding_item/embeddings/m/Read/ReadVariableOp7Adam/mf_embedding_user/embeddings/m/Read/ReadVariableOp7Adam/mf_embedding_item/embeddings/m/Read/ReadVariableOp(Adam/layer1/kernel/m/Read/ReadVariableOp&Adam/layer1/bias/m/Read/ReadVariableOp(Adam/layer2/kernel/m/Read/ReadVariableOp&Adam/layer2/bias/m/Read/ReadVariableOp(Adam/layer3/kernel/m/Read/ReadVariableOp&Adam/layer3/bias/m/Read/ReadVariableOp,Adam/prediction/kernel/m/Read/ReadVariableOp*Adam/prediction/bias/m/Read/ReadVariableOp8Adam/mlp_embedding_user/embeddings/v/Read/ReadVariableOp8Adam/mlp_embedding_item/embeddings/v/Read/ReadVariableOp7Adam/mf_embedding_user/embeddings/v/Read/ReadVariableOp7Adam/mf_embedding_item/embeddings/v/Read/ReadVariableOp(Adam/layer1/kernel/v/Read/ReadVariableOp&Adam/layer1/bias/v/Read/ReadVariableOp(Adam/layer2/kernel/v/Read/ReadVariableOp&Adam/layer2/bias/v/Read/ReadVariableOp(Adam/layer3/kernel/v/Read/ReadVariableOp&Adam/layer3/bias/v/Read/ReadVariableOp,Adam/prediction/kernel/v/Read/ReadVariableOp*Adam/prediction/bias/v/Read/ReadVariableOpConst*8
Tin1
/2-	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*(
f#R!
__inference__traced_save_876911
?

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemlp_embedding_user/embeddingsmlp_embedding_item/embeddingsmf_embedding_user/embeddingsmf_embedding_item/embeddingslayer1/kernellayer1/biaslayer2/kernellayer2/biaslayer3/kernellayer3/biasprediction/kernelprediction/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcount$Adam/mlp_embedding_user/embeddings/m$Adam/mlp_embedding_item/embeddings/m#Adam/mf_embedding_user/embeddings/m#Adam/mf_embedding_item/embeddings/mAdam/layer1/kernel/mAdam/layer1/bias/mAdam/layer2/kernel/mAdam/layer2/bias/mAdam/layer3/kernel/mAdam/layer3/bias/mAdam/prediction/kernel/mAdam/prediction/bias/m$Adam/mlp_embedding_user/embeddings/v$Adam/mlp_embedding_item/embeddings/v#Adam/mf_embedding_user/embeddings/v#Adam/mf_embedding_item/embeddings/vAdam/layer1/kernel/vAdam/layer1/bias/vAdam/layer2/kernel/vAdam/layer2/bias/vAdam/layer3/kernel/vAdam/layer3/bias/vAdam/prediction/kernel/vAdam/prediction/bias/v*7
Tin0
.2,*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*+
f&R$
"__inference__traced_restore_877052??	
?

?
B__inference_layer2_layer_call_and_return_conditional_losses_875998

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
layer2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
layer2/kernel/Regularizer/Const?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?	
?
M__inference_mf_embedding_user_layer_call_and_return_conditional_losses_875976

inputs
embedding_lookup_875969
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_875969inputs*
Tindices0**
_class 
loc:@embedding_lookup/875969*+
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/875969*+
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2
embedding_lookup/Identity_1?
.mf_embedding_user/embeddings/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    20
.mf_embedding_user/embeddings/Regularizer/Const?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:& "
 
_user_specified_nameinputs
?
,
__inference_loss_fn_6_876757
identity?
layer3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
layer3/kernel/Regularizer/Constk
IdentityIdentity(layer3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_876030

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?L
?
A__inference_model_layer_call_and_return_conditional_losses_876124

user_input

item_input5
1mlp_embedding_item_statefulpartitionedcall_args_15
1mlp_embedding_user_statefulpartitionedcall_args_1)
%layer1_statefulpartitionedcall_args_1)
%layer1_statefulpartitionedcall_args_24
0mf_embedding_item_statefulpartitionedcall_args_14
0mf_embedding_user_statefulpartitionedcall_args_1)
%layer2_statefulpartitionedcall_args_1)
%layer2_statefulpartitionedcall_args_2)
%layer3_statefulpartitionedcall_args_1)
%layer3_statefulpartitionedcall_args_2-
)prediction_statefulpartitionedcall_args_1-
)prediction_statefulpartitionedcall_args_2
identity??layer1/StatefulPartitionedCall?layer2/StatefulPartitionedCall?layer3/StatefulPartitionedCall?)mf_embedding_item/StatefulPartitionedCall?)mf_embedding_user/StatefulPartitionedCall?*mlp_embedding_item/StatefulPartitionedCall?*mlp_embedding_user/StatefulPartitionedCall?"prediction/StatefulPartitionedCall?
*mlp_embedding_item/StatefulPartitionedCallStatefulPartitionedCall
item_input1mlp_embedding_item_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:????????? *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*W
fRRP
N__inference_mlp_embedding_item_layer_call_and_return_conditional_losses_8758482,
*mlp_embedding_item/StatefulPartitionedCall?
*mlp_embedding_user/StatefulPartitionedCallStatefulPartitionedCall
user_input1mlp_embedding_user_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:????????? *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*W
fRRP
N__inference_mlp_embedding_user_layer_call_and_return_conditional_losses_8758682,
*mlp_embedding_user/StatefulPartitionedCall?
flatten_2/PartitionedCallPartitionedCall3mlp_embedding_user/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:????????? *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_8758842
flatten_2/PartitionedCall?
flatten_3/PartitionedCallPartitionedCall3mlp_embedding_item/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:????????? *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_8758982
flatten_3/PartitionedCall?
concatenate/PartitionedCallPartitionedCall"flatten_2/PartitionedCall:output:0"flatten_3/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????@*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_8759132
concatenate/PartitionedCall?
layer1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0%layer1_statefulpartitionedcall_args_1%layer1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:????????? *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*K
fFRD
B__inference_layer1_layer_call_and_return_conditional_losses_8759342 
layer1/StatefulPartitionedCall?
)mf_embedding_item/StatefulPartitionedCallStatefulPartitionedCall
item_input0mf_embedding_item_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*V
fQRO
M__inference_mf_embedding_item_layer_call_and_return_conditional_losses_8759562+
)mf_embedding_item/StatefulPartitionedCall?
)mf_embedding_user/StatefulPartitionedCallStatefulPartitionedCall
user_input0mf_embedding_user_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*V
fQRO
M__inference_mf_embedding_user_layer_call_and_return_conditional_losses_8759762+
)mf_embedding_user/StatefulPartitionedCall?
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0%layer2_statefulpartitionedcall_args_1%layer2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*K
fFRD
B__inference_layer2_layer_call_and_return_conditional_losses_8759982 
layer2/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall2mf_embedding_user/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_8760162
flatten/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall2mf_embedding_item/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_8760302
flatten_1/PartitionedCall?
multiply/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*M
fHRF
D__inference_multiply_layer_call_and_return_conditional_losses_8760442
multiply/PartitionedCall?
layer3/StatefulPartitionedCallStatefulPartitionedCall'layer2/StatefulPartitionedCall:output:0%layer3_statefulpartitionedcall_args_1%layer3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*K
fFRD
B__inference_layer3_layer_call_and_return_conditional_losses_8760652 
layer3/StatefulPartitionedCall?
concatenate_1/PartitionedCallPartitionedCall!multiply/PartitionedCall:output:0'layer3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_8760842
concatenate_1/PartitionedCall?
"prediction/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0)prediction_statefulpartitionedcall_args_1)prediction_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*O
fJRH
F__inference_prediction_layer_call_and_return_conditional_losses_8761042$
"prediction/StatefulPartitionedCall?
/mlp_embedding_user/embeddings/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    21
/mlp_embedding_user/embeddings/Regularizer/Const?
/mlp_embedding_item/embeddings/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    21
/mlp_embedding_item/embeddings/Regularizer/Const?
.mf_embedding_user/embeddings/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    20
.mf_embedding_user/embeddings/Regularizer/Const?
.mf_embedding_item/embeddings/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    20
.mf_embedding_item/embeddings/Regularizer/Const?
layer1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
layer1/kernel/Regularizer/Const?
layer2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
layer2/kernel/Regularizer/Const?
layer3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
layer3/kernel/Regularizer/Const?
IdentityIdentity+prediction/StatefulPartitionedCall:output:0^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall^layer3/StatefulPartitionedCall*^mf_embedding_item/StatefulPartitionedCall*^mf_embedding_user/StatefulPartitionedCall+^mlp_embedding_item/StatefulPartitionedCall+^mlp_embedding_user/StatefulPartitionedCall#^prediction/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:?????????:?????????::::::::::::2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2@
layer3/StatefulPartitionedCalllayer3/StatefulPartitionedCall2V
)mf_embedding_item/StatefulPartitionedCall)mf_embedding_item/StatefulPartitionedCall2V
)mf_embedding_user/StatefulPartitionedCall)mf_embedding_user/StatefulPartitionedCall2X
*mlp_embedding_item/StatefulPartitionedCall*mlp_embedding_item/StatefulPartitionedCall2X
*mlp_embedding_user/StatefulPartitionedCall*mlp_embedding_user/StatefulPartitionedCall2H
"prediction/StatefulPartitionedCall"prediction/StatefulPartitionedCall:* &
$
_user_specified_name
user_input:*&
$
_user_specified_name
item_input
?
,
__inference_loss_fn_4_876747
identity?
layer1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
layer1/kernel/Regularizer/Constk
IdentityIdentity(layer1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?\
?
A__inference_model_layer_call_and_return_conditional_losses_876385
inputs_0
inputs_1.
*mlp_embedding_item_embedding_lookup_876316.
*mlp_embedding_user_embedding_lookup_876321)
%layer1_matmul_readvariableop_resource*
&layer1_biasadd_readvariableop_resource-
)mf_embedding_item_embedding_lookup_876339-
)mf_embedding_user_embedding_lookup_876344)
%layer2_matmul_readvariableop_resource*
&layer2_biasadd_readvariableop_resource)
%layer3_matmul_readvariableop_resource*
&layer3_biasadd_readvariableop_resource-
)prediction_matmul_readvariableop_resource.
*prediction_biasadd_readvariableop_resource
identity??layer1/BiasAdd/ReadVariableOp?layer1/MatMul/ReadVariableOp?layer2/BiasAdd/ReadVariableOp?layer2/MatMul/ReadVariableOp?layer3/BiasAdd/ReadVariableOp?layer3/MatMul/ReadVariableOp?"mf_embedding_item/embedding_lookup?"mf_embedding_user/embedding_lookup?#mlp_embedding_item/embedding_lookup?#mlp_embedding_user/embedding_lookup?!prediction/BiasAdd/ReadVariableOp? prediction/MatMul/ReadVariableOp?
#mlp_embedding_item/embedding_lookupResourceGather*mlp_embedding_item_embedding_lookup_876316inputs_1*
Tindices0*=
_class3
1/loc:@mlp_embedding_item/embedding_lookup/876316*+
_output_shapes
:????????? *
dtype02%
#mlp_embedding_item/embedding_lookup?
,mlp_embedding_item/embedding_lookup/IdentityIdentity,mlp_embedding_item/embedding_lookup:output:0*
T0*=
_class3
1/loc:@mlp_embedding_item/embedding_lookup/876316*+
_output_shapes
:????????? 2.
,mlp_embedding_item/embedding_lookup/Identity?
.mlp_embedding_item/embedding_lookup/Identity_1Identity5mlp_embedding_item/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:????????? 20
.mlp_embedding_item/embedding_lookup/Identity_1?
#mlp_embedding_user/embedding_lookupResourceGather*mlp_embedding_user_embedding_lookup_876321inputs_0*
Tindices0*=
_class3
1/loc:@mlp_embedding_user/embedding_lookup/876321*+
_output_shapes
:????????? *
dtype02%
#mlp_embedding_user/embedding_lookup?
,mlp_embedding_user/embedding_lookup/IdentityIdentity,mlp_embedding_user/embedding_lookup:output:0*
T0*=
_class3
1/loc:@mlp_embedding_user/embedding_lookup/876321*+
_output_shapes
:????????? 2.
,mlp_embedding_user/embedding_lookup/Identity?
.mlp_embedding_user/embedding_lookup/Identity_1Identity5mlp_embedding_user/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:????????? 20
.mlp_embedding_user/embedding_lookup/Identity_1s
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"????    2
flatten_2/Const?
flatten_2/ReshapeReshape7mlp_embedding_user/embedding_lookup/Identity_1:output:0flatten_2/Const:output:0*
T0*'
_output_shapes
:????????? 2
flatten_2/Reshapes
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"????    2
flatten_3/Const?
flatten_3/ReshapeReshape7mlp_embedding_item/embedding_lookup/Identity_1:output:0flatten_3/Const:output:0*
T0*'
_output_shapes
:????????? 2
flatten_3/Reshapet
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2flatten_2/Reshape:output:0flatten_3/Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????@2
concatenate/concat?
layer1/MatMul/ReadVariableOpReadVariableOp%layer1_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
layer1/MatMul/ReadVariableOp?
layer1/MatMulMatMulconcatenate/concat:output:0$layer1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
layer1/MatMul?
layer1/BiasAdd/ReadVariableOpReadVariableOp&layer1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
layer1/BiasAdd/ReadVariableOp?
layer1/BiasAddBiasAddlayer1/MatMul:product:0%layer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
layer1/BiasAddm
layer1/ReluRelulayer1/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
layer1/Relu?
"mf_embedding_item/embedding_lookupResourceGather)mf_embedding_item_embedding_lookup_876339inputs_1*
Tindices0*<
_class2
0.loc:@mf_embedding_item/embedding_lookup/876339*+
_output_shapes
:?????????*
dtype02$
"mf_embedding_item/embedding_lookup?
+mf_embedding_item/embedding_lookup/IdentityIdentity+mf_embedding_item/embedding_lookup:output:0*
T0*<
_class2
0.loc:@mf_embedding_item/embedding_lookup/876339*+
_output_shapes
:?????????2-
+mf_embedding_item/embedding_lookup/Identity?
-mf_embedding_item/embedding_lookup/Identity_1Identity4mf_embedding_item/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2/
-mf_embedding_item/embedding_lookup/Identity_1?
"mf_embedding_user/embedding_lookupResourceGather)mf_embedding_user_embedding_lookup_876344inputs_0*
Tindices0*<
_class2
0.loc:@mf_embedding_user/embedding_lookup/876344*+
_output_shapes
:?????????*
dtype02$
"mf_embedding_user/embedding_lookup?
+mf_embedding_user/embedding_lookup/IdentityIdentity+mf_embedding_user/embedding_lookup:output:0*
T0*<
_class2
0.loc:@mf_embedding_user/embedding_lookup/876344*+
_output_shapes
:?????????2-
+mf_embedding_user/embedding_lookup/Identity?
-mf_embedding_user/embedding_lookup/Identity_1Identity4mf_embedding_user/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2/
-mf_embedding_user/embedding_lookup/Identity_1?
layer2/MatMul/ReadVariableOpReadVariableOp%layer2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
layer2/MatMul/ReadVariableOp?
layer2/MatMulMatMullayer1/Relu:activations:0$layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
layer2/MatMul?
layer2/BiasAdd/ReadVariableOpReadVariableOp&layer2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
layer2/BiasAdd/ReadVariableOp?
layer2/BiasAddBiasAddlayer2/MatMul:product:0%layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
layer2/BiasAddm
layer2/ReluRelulayer2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
layer2/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Const?
flatten/ReshapeReshape6mf_embedding_user/embedding_lookup/Identity_1:output:0flatten/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten/Reshapes
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_1/Const?
flatten_1/ReshapeReshape6mf_embedding_item/embedding_lookup/Identity_1:output:0flatten_1/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_1/Reshape?
multiply/mulMulflatten/Reshape:output:0flatten_1/Reshape:output:0*
T0*'
_output_shapes
:?????????2
multiply/mul?
layer3/MatMul/ReadVariableOpReadVariableOp%layer3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
layer3/MatMul/ReadVariableOp?
layer3/MatMulMatMullayer2/Relu:activations:0$layer3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
layer3/MatMul?
layer3/BiasAdd/ReadVariableOpReadVariableOp&layer3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
layer3/BiasAdd/ReadVariableOp?
layer3/BiasAddBiasAddlayer3/MatMul:product:0%layer3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
layer3/BiasAddm
layer3/ReluRelulayer3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
layer3/Relux
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2multiply/mul:z:0layer3/Relu:activations:0"concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatenate_1/concat?
 prediction/MatMul/ReadVariableOpReadVariableOp)prediction_matmul_readvariableop_resource*
_output_shapes

:*
dtype02"
 prediction/MatMul/ReadVariableOp?
prediction/MatMulMatMulconcatenate_1/concat:output:0(prediction/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
prediction/MatMul?
!prediction/BiasAdd/ReadVariableOpReadVariableOp*prediction_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!prediction/BiasAdd/ReadVariableOp?
prediction/BiasAddBiasAddprediction/MatMul:product:0)prediction/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
prediction/BiasAdd?
prediction/SigmoidSigmoidprediction/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
prediction/Sigmoid?
/mlp_embedding_user/embeddings/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    21
/mlp_embedding_user/embeddings/Regularizer/Const?
/mlp_embedding_item/embeddings/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    21
/mlp_embedding_item/embeddings/Regularizer/Const?
.mf_embedding_user/embeddings/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    20
.mf_embedding_user/embeddings/Regularizer/Const?
.mf_embedding_item/embeddings/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    20
.mf_embedding_item/embeddings/Regularizer/Const?
layer1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
layer1/kernel/Regularizer/Const?
layer2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
layer2/kernel/Regularizer/Const?
layer3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
layer3/kernel/Regularizer/Const?
IdentityIdentityprediction/Sigmoid:y:0^layer1/BiasAdd/ReadVariableOp^layer1/MatMul/ReadVariableOp^layer2/BiasAdd/ReadVariableOp^layer2/MatMul/ReadVariableOp^layer3/BiasAdd/ReadVariableOp^layer3/MatMul/ReadVariableOp#^mf_embedding_item/embedding_lookup#^mf_embedding_user/embedding_lookup$^mlp_embedding_item/embedding_lookup$^mlp_embedding_user/embedding_lookup"^prediction/BiasAdd/ReadVariableOp!^prediction/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:?????????:?????????::::::::::::2>
layer1/BiasAdd/ReadVariableOplayer1/BiasAdd/ReadVariableOp2<
layer1/MatMul/ReadVariableOplayer1/MatMul/ReadVariableOp2>
layer2/BiasAdd/ReadVariableOplayer2/BiasAdd/ReadVariableOp2<
layer2/MatMul/ReadVariableOplayer2/MatMul/ReadVariableOp2>
layer3/BiasAdd/ReadVariableOplayer3/BiasAdd/ReadVariableOp2<
layer3/MatMul/ReadVariableOplayer3/MatMul/ReadVariableOp2H
"mf_embedding_item/embedding_lookup"mf_embedding_item/embedding_lookup2H
"mf_embedding_user/embedding_lookup"mf_embedding_user/embedding_lookup2J
#mlp_embedding_item/embedding_lookup#mlp_embedding_item/embedding_lookup2J
#mlp_embedding_user/embedding_lookup#mlp_embedding_user/embedding_lookup2F
!prediction/BiasAdd/ReadVariableOp!prediction/BiasAdd/ReadVariableOp2D
 prediction/MatMul/ReadVariableOp prediction/MatMul/ReadVariableOp:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?
q
G__inference_concatenate_layer_call_and_return_conditional_losses_875913

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????@2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:????????? :????????? :& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
?
?
2__inference_mf_embedding_item_layer_call_fn_876597

inputs"
statefulpartitionedcall_args_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*V
fQRO
M__inference_mf_embedding_item_layer_call_and_return_conditional_losses_8759562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?	
?
M__inference_mf_embedding_user_layer_call_and_return_conditional_losses_876574

inputs
embedding_lookup_876567
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_876567inputs*
Tindices0**
_class 
loc:@embedding_lookup/876567*+
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/876567*+
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2
embedding_lookup/Identity_1?
.mf_embedding_user/embeddings/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    20
.mf_embedding_user/embeddings/Regularizer/Const?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:& "
 
_user_specified_nameinputs
?
_
C__inference_flatten_layer_call_and_return_conditional_losses_876016

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?	
?
F__inference_prediction_layer_call_and_return_conditional_losses_876104

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_876545

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????    2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:????????? 2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0**
_input_shapes
:????????? :& "
 
_user_specified_nameinputs
?

?
B__inference_layer3_layer_call_and_return_conditional_losses_876684

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
layer3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
layer3/kernel/Regularizer/Const?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
&__inference_model_layer_call_fn_876221

user_input

item_input"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
user_input
item_inputstatefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_8762062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:?????????:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:* &
$
_user_specified_name
user_input:*&
$
_user_specified_name
item_input
?
F
*__inference_flatten_2_layer_call_fn_876539

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:????????? *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_8758842
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0**
_input_shapes
:????????? :& "
 
_user_specified_nameinputs
?	
?
N__inference_mlp_embedding_user_layer_call_and_return_conditional_losses_875868

inputs
embedding_lookup_875861
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_875861inputs*
Tindices0**
_class 
loc:@embedding_lookup/875861*+
_output_shapes
:????????? *
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/875861*+
_output_shapes
:????????? 2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:????????? 2
embedding_lookup/Identity_1?
/mlp_embedding_user/embeddings/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    21
/mlp_embedding_user/embeddings/Regularizer/Const?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:& "
 
_user_specified_nameinputs
?
,
__inference_loss_fn_3_876742
identity?
.mf_embedding_item/embeddings/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    20
.mf_embedding_item/embeddings/Regularizer/Constz
IdentityIdentity7mf_embedding_item/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?	
?
N__inference_mlp_embedding_user_layer_call_and_return_conditional_losses_876505

inputs
embedding_lookup_876498
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_876498inputs*
Tindices0**
_class 
loc:@embedding_lookup/876498*+
_output_shapes
:????????? *
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/876498*+
_output_shapes
:????????? 2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:????????? 2
embedding_lookup/Identity_1?
/mlp_embedding_user/embeddings/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    21
/mlp_embedding_user/embeddings/Regularizer/Const?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:& "
 
_user_specified_nameinputs
?
s
I__inference_concatenate_1_layer_call_and_return_conditional_losses_876084

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
?K
?
A__inference_model_layer_call_and_return_conditional_losses_876263

inputs
inputs_15
1mlp_embedding_item_statefulpartitionedcall_args_15
1mlp_embedding_user_statefulpartitionedcall_args_1)
%layer1_statefulpartitionedcall_args_1)
%layer1_statefulpartitionedcall_args_24
0mf_embedding_item_statefulpartitionedcall_args_14
0mf_embedding_user_statefulpartitionedcall_args_1)
%layer2_statefulpartitionedcall_args_1)
%layer2_statefulpartitionedcall_args_2)
%layer3_statefulpartitionedcall_args_1)
%layer3_statefulpartitionedcall_args_2-
)prediction_statefulpartitionedcall_args_1-
)prediction_statefulpartitionedcall_args_2
identity??layer1/StatefulPartitionedCall?layer2/StatefulPartitionedCall?layer3/StatefulPartitionedCall?)mf_embedding_item/StatefulPartitionedCall?)mf_embedding_user/StatefulPartitionedCall?*mlp_embedding_item/StatefulPartitionedCall?*mlp_embedding_user/StatefulPartitionedCall?"prediction/StatefulPartitionedCall?
*mlp_embedding_item/StatefulPartitionedCallStatefulPartitionedCallinputs_11mlp_embedding_item_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:????????? *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*W
fRRP
N__inference_mlp_embedding_item_layer_call_and_return_conditional_losses_8758482,
*mlp_embedding_item/StatefulPartitionedCall?
*mlp_embedding_user/StatefulPartitionedCallStatefulPartitionedCallinputs1mlp_embedding_user_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:????????? *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*W
fRRP
N__inference_mlp_embedding_user_layer_call_and_return_conditional_losses_8758682,
*mlp_embedding_user/StatefulPartitionedCall?
flatten_2/PartitionedCallPartitionedCall3mlp_embedding_user/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:????????? *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_8758842
flatten_2/PartitionedCall?
flatten_3/PartitionedCallPartitionedCall3mlp_embedding_item/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:????????? *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_8758982
flatten_3/PartitionedCall?
concatenate/PartitionedCallPartitionedCall"flatten_2/PartitionedCall:output:0"flatten_3/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????@*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_8759132
concatenate/PartitionedCall?
layer1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0%layer1_statefulpartitionedcall_args_1%layer1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:????????? *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*K
fFRD
B__inference_layer1_layer_call_and_return_conditional_losses_8759342 
layer1/StatefulPartitionedCall?
)mf_embedding_item/StatefulPartitionedCallStatefulPartitionedCallinputs_10mf_embedding_item_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*V
fQRO
M__inference_mf_embedding_item_layer_call_and_return_conditional_losses_8759562+
)mf_embedding_item/StatefulPartitionedCall?
)mf_embedding_user/StatefulPartitionedCallStatefulPartitionedCallinputs0mf_embedding_user_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*V
fQRO
M__inference_mf_embedding_user_layer_call_and_return_conditional_losses_8759762+
)mf_embedding_user/StatefulPartitionedCall?
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0%layer2_statefulpartitionedcall_args_1%layer2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*K
fFRD
B__inference_layer2_layer_call_and_return_conditional_losses_8759982 
layer2/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall2mf_embedding_user/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_8760162
flatten/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall2mf_embedding_item/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_8760302
flatten_1/PartitionedCall?
multiply/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*M
fHRF
D__inference_multiply_layer_call_and_return_conditional_losses_8760442
multiply/PartitionedCall?
layer3/StatefulPartitionedCallStatefulPartitionedCall'layer2/StatefulPartitionedCall:output:0%layer3_statefulpartitionedcall_args_1%layer3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*K
fFRD
B__inference_layer3_layer_call_and_return_conditional_losses_8760652 
layer3/StatefulPartitionedCall?
concatenate_1/PartitionedCallPartitionedCall!multiply/PartitionedCall:output:0'layer3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_8760842
concatenate_1/PartitionedCall?
"prediction/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0)prediction_statefulpartitionedcall_args_1)prediction_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*O
fJRH
F__inference_prediction_layer_call_and_return_conditional_losses_8761042$
"prediction/StatefulPartitionedCall?
/mlp_embedding_user/embeddings/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    21
/mlp_embedding_user/embeddings/Regularizer/Const?
/mlp_embedding_item/embeddings/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    21
/mlp_embedding_item/embeddings/Regularizer/Const?
.mf_embedding_user/embeddings/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    20
.mf_embedding_user/embeddings/Regularizer/Const?
.mf_embedding_item/embeddings/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    20
.mf_embedding_item/embeddings/Regularizer/Const?
layer1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
layer1/kernel/Regularizer/Const?
layer2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
layer2/kernel/Regularizer/Const?
layer3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
layer3/kernel/Regularizer/Const?
IdentityIdentity+prediction/StatefulPartitionedCall:output:0^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall^layer3/StatefulPartitionedCall*^mf_embedding_item/StatefulPartitionedCall*^mf_embedding_user/StatefulPartitionedCall+^mlp_embedding_item/StatefulPartitionedCall+^mlp_embedding_user/StatefulPartitionedCall#^prediction/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:?????????:?????????::::::::::::2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2@
layer3/StatefulPartitionedCalllayer3/StatefulPartitionedCall2V
)mf_embedding_item/StatefulPartitionedCall)mf_embedding_item/StatefulPartitionedCall2V
)mf_embedding_user/StatefulPartitionedCall)mf_embedding_user/StatefulPartitionedCall2X
*mlp_embedding_item/StatefulPartitionedCall*mlp_embedding_item/StatefulPartitionedCall2X
*mlp_embedding_user/StatefulPartitionedCall*mlp_embedding_user/StatefulPartitionedCall2H
"prediction/StatefulPartitionedCall"prediction/StatefulPartitionedCall:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
?
s
G__inference_concatenate_layer_call_and_return_conditional_losses_876557
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????@2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:????????? :????????? :( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?
?
'__inference_layer3_layer_call_fn_876691

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*K
fFRD
B__inference_layer3_layer_call_and_return_conditional_losses_8760652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
,
__inference_loss_fn_2_876737
identity?
.mf_embedding_user/embeddings/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    20
.mf_embedding_user/embeddings/Regularizer/Constz
IdentityIdentity7mf_embedding_user/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?U
?
__inference__traced_save_876911
file_prefix<
8savev2_mlp_embedding_user_embeddings_read_readvariableop<
8savev2_mlp_embedding_item_embeddings_read_readvariableop;
7savev2_mf_embedding_user_embeddings_read_readvariableop;
7savev2_mf_embedding_item_embeddings_read_readvariableop,
(savev2_layer1_kernel_read_readvariableop*
&savev2_layer1_bias_read_readvariableop,
(savev2_layer2_kernel_read_readvariableop*
&savev2_layer2_bias_read_readvariableop,
(savev2_layer3_kernel_read_readvariableop*
&savev2_layer3_bias_read_readvariableop0
,savev2_prediction_kernel_read_readvariableop.
*savev2_prediction_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopC
?savev2_adam_mlp_embedding_user_embeddings_m_read_readvariableopC
?savev2_adam_mlp_embedding_item_embeddings_m_read_readvariableopB
>savev2_adam_mf_embedding_user_embeddings_m_read_readvariableopB
>savev2_adam_mf_embedding_item_embeddings_m_read_readvariableop3
/savev2_adam_layer1_kernel_m_read_readvariableop1
-savev2_adam_layer1_bias_m_read_readvariableop3
/savev2_adam_layer2_kernel_m_read_readvariableop1
-savev2_adam_layer2_bias_m_read_readvariableop3
/savev2_adam_layer3_kernel_m_read_readvariableop1
-savev2_adam_layer3_bias_m_read_readvariableop7
3savev2_adam_prediction_kernel_m_read_readvariableop5
1savev2_adam_prediction_bias_m_read_readvariableopC
?savev2_adam_mlp_embedding_user_embeddings_v_read_readvariableopC
?savev2_adam_mlp_embedding_item_embeddings_v_read_readvariableopB
>savev2_adam_mf_embedding_user_embeddings_v_read_readvariableopB
>savev2_adam_mf_embedding_item_embeddings_v_read_readvariableop3
/savev2_adam_layer1_kernel_v_read_readvariableop1
-savev2_adam_layer1_bias_v_read_readvariableop3
/savev2_adam_layer2_kernel_v_read_readvariableop1
-savev2_adam_layer2_bias_v_read_readvariableop3
/savev2_adam_layer3_kernel_v_read_readvariableop1
-savev2_adam_layer3_bias_v_read_readvariableop7
3savev2_adam_prediction_kernel_v_read_readvariableop5
1savev2_adam_prediction_bias_v_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_95ec1a50b89c48b5bbb7f3bf947c7b5c/part2
StringJoin/inputs_1?

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*?
value?B?+B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:08savev2_mlp_embedding_user_embeddings_read_readvariableop8savev2_mlp_embedding_item_embeddings_read_readvariableop7savev2_mf_embedding_user_embeddings_read_readvariableop7savev2_mf_embedding_item_embeddings_read_readvariableop(savev2_layer1_kernel_read_readvariableop&savev2_layer1_bias_read_readvariableop(savev2_layer2_kernel_read_readvariableop&savev2_layer2_bias_read_readvariableop(savev2_layer3_kernel_read_readvariableop&savev2_layer3_bias_read_readvariableop,savev2_prediction_kernel_read_readvariableop*savev2_prediction_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop?savev2_adam_mlp_embedding_user_embeddings_m_read_readvariableop?savev2_adam_mlp_embedding_item_embeddings_m_read_readvariableop>savev2_adam_mf_embedding_user_embeddings_m_read_readvariableop>savev2_adam_mf_embedding_item_embeddings_m_read_readvariableop/savev2_adam_layer1_kernel_m_read_readvariableop-savev2_adam_layer1_bias_m_read_readvariableop/savev2_adam_layer2_kernel_m_read_readvariableop-savev2_adam_layer2_bias_m_read_readvariableop/savev2_adam_layer3_kernel_m_read_readvariableop-savev2_adam_layer3_bias_m_read_readvariableop3savev2_adam_prediction_kernel_m_read_readvariableop1savev2_adam_prediction_bias_m_read_readvariableop?savev2_adam_mlp_embedding_user_embeddings_v_read_readvariableop?savev2_adam_mlp_embedding_item_embeddings_v_read_readvariableop>savev2_adam_mf_embedding_user_embeddings_v_read_readvariableop>savev2_adam_mf_embedding_item_embeddings_v_read_readvariableop/savev2_adam_layer1_kernel_v_read_readvariableop-savev2_adam_layer1_bias_v_read_readvariableop/savev2_adam_layer2_kernel_v_read_readvariableop-savev2_adam_layer2_bias_v_read_readvariableop/savev2_adam_layer3_kernel_v_read_readvariableop-savev2_adam_layer3_bias_v_read_readvariableop3savev2_adam_prediction_kernel_v_read_readvariableop1savev2_adam_prediction_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *9
dtypes/
-2+	2
SaveV2?
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1?
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names?
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity?

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :	?7 :	? :	?7:	?:@ : : :::::: : : : : : : :	?7 :	? :	?7:	?:@ : : ::::::	?7 :	? :	?7:	?:@ : : :::::: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
?
?
'__inference_layer1_layer_call_fn_876617

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:????????? *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*K
fFRD
B__inference_layer1_layer_call_and_return_conditional_losses_8759342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?	
?
N__inference_mlp_embedding_item_layer_call_and_return_conditional_losses_876522

inputs
embedding_lookup_876515
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_876515inputs*
Tindices0**
_class 
loc:@embedding_lookup/876515*+
_output_shapes
:????????? *
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/876515*+
_output_shapes
:????????? 2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:????????? 2
embedding_lookup/Identity_1?
/mlp_embedding_item/embeddings/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    21
/mlp_embedding_item/embeddings/Regularizer/Const?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:& "
 
_user_specified_nameinputs
?
?
&__inference_model_layer_call_fn_876494
inputs_0
inputs_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_8762632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:?????????:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?
U
)__inference_multiply_layer_call_fn_876671
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*M
fHRF
D__inference_multiply_layer_call_and_return_conditional_losses_8760442
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?
_
C__inference_flatten_layer_call_and_return_conditional_losses_876623

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?	
?
M__inference_mf_embedding_item_layer_call_and_return_conditional_losses_875956

inputs
embedding_lookup_875949
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_875949inputs*
Tindices0**
_class 
loc:@embedding_lookup/875949*+
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/875949*+
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2
embedding_lookup/Identity_1?
.mf_embedding_item/embeddings/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    20
.mf_embedding_item/embeddings/Regularizer/Const?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:& "
 
_user_specified_nameinputs
?
p
D__inference_multiply_layer_call_and_return_conditional_losses_876665
inputs_0
inputs_1
identityW
mulMulinputs_0inputs_1*
T0*'
_output_shapes
:?????????2
mul[
IdentityIdentitymul:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?	
?
F__inference_prediction_layer_call_and_return_conditional_losses_876715

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
'__inference_layer2_layer_call_fn_876659

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*K
fFRD
B__inference_layer2_layer_call_and_return_conditional_losses_8759982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
Z
.__inference_concatenate_1_layer_call_fn_876704
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_8760842
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?
?
3__inference_mlp_embedding_item_layer_call_fn_876528

inputs"
statefulpartitionedcall_args_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:????????? *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*W
fRRP
N__inference_mlp_embedding_item_layer_call_and_return_conditional_losses_8758482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
,
__inference_loss_fn_1_876732
identity?
/mlp_embedding_item/embeddings/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    21
/mlp_embedding_item/embeddings/Regularizer/Const{
IdentityIdentity8mlp_embedding_item/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?
a
E__inference_flatten_2_layer_call_and_return_conditional_losses_875884

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????    2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:????????? 2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0**
_input_shapes
:????????? :& "
 
_user_specified_nameinputs
?
F
*__inference_flatten_3_layer_call_fn_876550

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:????????? *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_8758982
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0**
_input_shapes
:????????? :& "
 
_user_specified_nameinputs
?K
?
A__inference_model_layer_call_and_return_conditional_losses_876206

inputs
inputs_15
1mlp_embedding_item_statefulpartitionedcall_args_15
1mlp_embedding_user_statefulpartitionedcall_args_1)
%layer1_statefulpartitionedcall_args_1)
%layer1_statefulpartitionedcall_args_24
0mf_embedding_item_statefulpartitionedcall_args_14
0mf_embedding_user_statefulpartitionedcall_args_1)
%layer2_statefulpartitionedcall_args_1)
%layer2_statefulpartitionedcall_args_2)
%layer3_statefulpartitionedcall_args_1)
%layer3_statefulpartitionedcall_args_2-
)prediction_statefulpartitionedcall_args_1-
)prediction_statefulpartitionedcall_args_2
identity??layer1/StatefulPartitionedCall?layer2/StatefulPartitionedCall?layer3/StatefulPartitionedCall?)mf_embedding_item/StatefulPartitionedCall?)mf_embedding_user/StatefulPartitionedCall?*mlp_embedding_item/StatefulPartitionedCall?*mlp_embedding_user/StatefulPartitionedCall?"prediction/StatefulPartitionedCall?
*mlp_embedding_item/StatefulPartitionedCallStatefulPartitionedCallinputs_11mlp_embedding_item_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:????????? *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*W
fRRP
N__inference_mlp_embedding_item_layer_call_and_return_conditional_losses_8758482,
*mlp_embedding_item/StatefulPartitionedCall?
*mlp_embedding_user/StatefulPartitionedCallStatefulPartitionedCallinputs1mlp_embedding_user_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:????????? *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*W
fRRP
N__inference_mlp_embedding_user_layer_call_and_return_conditional_losses_8758682,
*mlp_embedding_user/StatefulPartitionedCall?
flatten_2/PartitionedCallPartitionedCall3mlp_embedding_user/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:????????? *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_8758842
flatten_2/PartitionedCall?
flatten_3/PartitionedCallPartitionedCall3mlp_embedding_item/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:????????? *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_8758982
flatten_3/PartitionedCall?
concatenate/PartitionedCallPartitionedCall"flatten_2/PartitionedCall:output:0"flatten_3/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????@*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_8759132
concatenate/PartitionedCall?
layer1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0%layer1_statefulpartitionedcall_args_1%layer1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:????????? *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*K
fFRD
B__inference_layer1_layer_call_and_return_conditional_losses_8759342 
layer1/StatefulPartitionedCall?
)mf_embedding_item/StatefulPartitionedCallStatefulPartitionedCallinputs_10mf_embedding_item_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*V
fQRO
M__inference_mf_embedding_item_layer_call_and_return_conditional_losses_8759562+
)mf_embedding_item/StatefulPartitionedCall?
)mf_embedding_user/StatefulPartitionedCallStatefulPartitionedCallinputs0mf_embedding_user_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*V
fQRO
M__inference_mf_embedding_user_layer_call_and_return_conditional_losses_8759762+
)mf_embedding_user/StatefulPartitionedCall?
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0%layer2_statefulpartitionedcall_args_1%layer2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*K
fFRD
B__inference_layer2_layer_call_and_return_conditional_losses_8759982 
layer2/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall2mf_embedding_user/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_8760162
flatten/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall2mf_embedding_item/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_8760302
flatten_1/PartitionedCall?
multiply/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*M
fHRF
D__inference_multiply_layer_call_and_return_conditional_losses_8760442
multiply/PartitionedCall?
layer3/StatefulPartitionedCallStatefulPartitionedCall'layer2/StatefulPartitionedCall:output:0%layer3_statefulpartitionedcall_args_1%layer3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*K
fFRD
B__inference_layer3_layer_call_and_return_conditional_losses_8760652 
layer3/StatefulPartitionedCall?
concatenate_1/PartitionedCallPartitionedCall!multiply/PartitionedCall:output:0'layer3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_8760842
concatenate_1/PartitionedCall?
"prediction/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0)prediction_statefulpartitionedcall_args_1)prediction_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*O
fJRH
F__inference_prediction_layer_call_and_return_conditional_losses_8761042$
"prediction/StatefulPartitionedCall?
/mlp_embedding_user/embeddings/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    21
/mlp_embedding_user/embeddings/Regularizer/Const?
/mlp_embedding_item/embeddings/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    21
/mlp_embedding_item/embeddings/Regularizer/Const?
.mf_embedding_user/embeddings/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    20
.mf_embedding_user/embeddings/Regularizer/Const?
.mf_embedding_item/embeddings/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    20
.mf_embedding_item/embeddings/Regularizer/Const?
layer1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
layer1/kernel/Regularizer/Const?
layer2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
layer2/kernel/Regularizer/Const?
layer3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
layer3/kernel/Regularizer/Const?
IdentityIdentity+prediction/StatefulPartitionedCall:output:0^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall^layer3/StatefulPartitionedCall*^mf_embedding_item/StatefulPartitionedCall*^mf_embedding_user/StatefulPartitionedCall+^mlp_embedding_item/StatefulPartitionedCall+^mlp_embedding_user/StatefulPartitionedCall#^prediction/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:?????????:?????????::::::::::::2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2@
layer3/StatefulPartitionedCalllayer3/StatefulPartitionedCall2V
)mf_embedding_item/StatefulPartitionedCall)mf_embedding_item/StatefulPartitionedCall2V
)mf_embedding_user/StatefulPartitionedCall)mf_embedding_user/StatefulPartitionedCall2X
*mlp_embedding_item/StatefulPartitionedCall*mlp_embedding_item/StatefulPartitionedCall2X
*mlp_embedding_user/StatefulPartitionedCall*mlp_embedding_user/StatefulPartitionedCall2H
"prediction/StatefulPartitionedCall"prediction/StatefulPartitionedCall:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
?	
?
N__inference_mlp_embedding_item_layer_call_and_return_conditional_losses_875848

inputs
embedding_lookup_875841
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_875841inputs*
Tindices0**
_class 
loc:@embedding_lookup/875841*+
_output_shapes
:????????? *
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/875841*+
_output_shapes
:????????? 2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:????????? 2
embedding_lookup/Identity_1?
/mlp_embedding_item/embeddings/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    21
/mlp_embedding_item/embeddings/Regularizer/Const?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:& "
 
_user_specified_nameinputs
?
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_876634

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
??
?
"__inference__traced_restore_877052
file_prefix2
.assignvariableop_mlp_embedding_user_embeddings4
0assignvariableop_1_mlp_embedding_item_embeddings3
/assignvariableop_2_mf_embedding_user_embeddings3
/assignvariableop_3_mf_embedding_item_embeddings$
 assignvariableop_4_layer1_kernel"
assignvariableop_5_layer1_bias$
 assignvariableop_6_layer2_kernel"
assignvariableop_7_layer2_bias$
 assignvariableop_8_layer3_kernel"
assignvariableop_9_layer3_bias)
%assignvariableop_10_prediction_kernel'
#assignvariableop_11_prediction_bias!
assignvariableop_12_adam_iter#
assignvariableop_13_adam_beta_1#
assignvariableop_14_adam_beta_2"
assignvariableop_15_adam_decay*
&assignvariableop_16_adam_learning_rate
assignvariableop_17_total
assignvariableop_18_count<
8assignvariableop_19_adam_mlp_embedding_user_embeddings_m<
8assignvariableop_20_adam_mlp_embedding_item_embeddings_m;
7assignvariableop_21_adam_mf_embedding_user_embeddings_m;
7assignvariableop_22_adam_mf_embedding_item_embeddings_m,
(assignvariableop_23_adam_layer1_kernel_m*
&assignvariableop_24_adam_layer1_bias_m,
(assignvariableop_25_adam_layer2_kernel_m*
&assignvariableop_26_adam_layer2_bias_m,
(assignvariableop_27_adam_layer3_kernel_m*
&assignvariableop_28_adam_layer3_bias_m0
,assignvariableop_29_adam_prediction_kernel_m.
*assignvariableop_30_adam_prediction_bias_m<
8assignvariableop_31_adam_mlp_embedding_user_embeddings_v<
8assignvariableop_32_adam_mlp_embedding_item_embeddings_v;
7assignvariableop_33_adam_mf_embedding_user_embeddings_v;
7assignvariableop_34_adam_mf_embedding_item_embeddings_v,
(assignvariableop_35_adam_layer1_kernel_v*
&assignvariableop_36_adam_layer1_bias_v,
(assignvariableop_37_adam_layer2_kernel_v*
&assignvariableop_38_adam_layer2_bias_v,
(assignvariableop_39_adam_layer3_kernel_v*
&assignvariableop_40_adam_layer3_bias_v0
,assignvariableop_41_adam_prediction_kernel_v.
*assignvariableop_42_adam_prediction_bias_v
identity_44??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*?
value?B?+B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::*9
dtypes/
-2+	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp.assignvariableop_mlp_embedding_user_embeddingsIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp0assignvariableop_1_mlp_embedding_item_embeddingsIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp/assignvariableop_2_mf_embedding_user_embeddingsIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp/assignvariableop_3_mf_embedding_item_embeddingsIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp assignvariableop_4_layer1_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_layer1_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp assignvariableop_6_layer2_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_layer2_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp assignvariableop_8_layer3_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_layer3_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp%assignvariableop_10_prediction_kernelIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp#assignvariableop_11_prediction_biasIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0	*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp8assignvariableop_19_adam_mlp_embedding_user_embeddings_mIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp8assignvariableop_20_adam_mlp_embedding_item_embeddings_mIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp7assignvariableop_21_adam_mf_embedding_user_embeddings_mIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp7assignvariableop_22_adam_mf_embedding_item_embeddings_mIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_layer1_kernel_mIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_layer1_bias_mIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_layer2_kernel_mIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp&assignvariableop_26_adam_layer2_bias_mIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_layer3_kernel_mIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp&assignvariableop_28_adam_layer3_bias_mIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp,assignvariableop_29_adam_prediction_kernel_mIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp*assignvariableop_30_adam_prediction_bias_mIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp8assignvariableop_31_adam_mlp_embedding_user_embeddings_vIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp8assignvariableop_32_adam_mlp_embedding_item_embeddings_vIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp7assignvariableop_33_adam_mf_embedding_user_embeddings_vIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp7assignvariableop_34_adam_mf_embedding_item_embeddings_vIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp(assignvariableop_35_adam_layer1_kernel_vIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp&assignvariableop_36_adam_layer1_bias_vIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp(assignvariableop_37_adam_layer2_kernel_vIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp&assignvariableop_38_adam_layer2_bias_vIdentity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp(assignvariableop_39_adam_layer3_kernel_vIdentity_39:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_39_
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp&assignvariableop_40_adam_layer3_bias_vIdentity_40:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_40_
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp,assignvariableop_41_adam_prediction_kernel_vIdentity_41:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_41_
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp*assignvariableop_42_adam_prediction_bias_vIdentity_42:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_42?
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names?
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_43Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_43?
Identity_44IdentityIdentity_43:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_44"#
identity_44Identity_44:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
?

?
B__inference_layer1_layer_call_and_return_conditional_losses_876610

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Relu?
layer1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
layer1/kernel/Regularizer/Const?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?	
?
M__inference_mf_embedding_item_layer_call_and_return_conditional_losses_876591

inputs
embedding_lookup_876584
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_876584inputs*
Tindices0**
_class 
loc:@embedding_lookup/876584*+
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/876584*+
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2
embedding_lookup/Identity_1?
.mf_embedding_item/embeddings/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    20
.mf_embedding_item/embeddings/Regularizer/Const?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:& "
 
_user_specified_nameinputs
?
?
2__inference_mf_embedding_user_layer_call_fn_876580

inputs"
statefulpartitionedcall_args_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*V
fQRO
M__inference_mf_embedding_user_layer_call_and_return_conditional_losses_8759762
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
,
__inference_loss_fn_0_876727
identity?
/mlp_embedding_user/embeddings/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    21
/mlp_embedding_user/embeddings/Regularizer/Const{
IdentityIdentity8mlp_embedding_user/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?
?
&__inference_model_layer_call_fn_876278

user_input

item_input"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
user_input
item_inputstatefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_8762632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:?????????:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:* &
$
_user_specified_name
user_input:*&
$
_user_specified_name
item_input
?
a
E__inference_flatten_2_layer_call_and_return_conditional_losses_876534

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????    2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:????????? 2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0**
_input_shapes
:????????? :& "
 
_user_specified_nameinputs
?
D
(__inference_flatten_layer_call_fn_876628

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_8760162
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?
F
*__inference_flatten_1_layer_call_fn_876639

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_8760302
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?

?
B__inference_layer3_layer_call_and_return_conditional_losses_876065

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
layer3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
layer3/kernel/Regularizer/Const?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?
X
,__inference_concatenate_layer_call_fn_876563
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????@*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_8759132
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:????????? :????????? :( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?

?
B__inference_layer2_layer_call_and_return_conditional_losses_876652

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
layer2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
layer2/kernel/Regularizer/Const?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?

?
B__inference_layer1_layer_call_and_return_conditional_losses_875934

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Relu?
layer1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
layer1/kernel/Regularizer/Const?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_876312

item_input

user_input"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
user_input
item_inputstatefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8**
f%R#
!__inference__wrapped_model_8758332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:?????????:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:* &
$
_user_specified_name
item_input:*&
$
_user_specified_name
user_input
?
n
D__inference_multiply_layer_call_and_return_conditional_losses_876044

inputs
inputs_1
identityU
mulMulinputsinputs_1*
T0*'
_output_shapes
:?????????2
mul[
IdentityIdentitymul:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
?
?
&__inference_model_layer_call_fn_876476
inputs_0
inputs_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_8762062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:?????????:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?L
?
A__inference_model_layer_call_and_return_conditional_losses_876163

user_input

item_input5
1mlp_embedding_item_statefulpartitionedcall_args_15
1mlp_embedding_user_statefulpartitionedcall_args_1)
%layer1_statefulpartitionedcall_args_1)
%layer1_statefulpartitionedcall_args_24
0mf_embedding_item_statefulpartitionedcall_args_14
0mf_embedding_user_statefulpartitionedcall_args_1)
%layer2_statefulpartitionedcall_args_1)
%layer2_statefulpartitionedcall_args_2)
%layer3_statefulpartitionedcall_args_1)
%layer3_statefulpartitionedcall_args_2-
)prediction_statefulpartitionedcall_args_1-
)prediction_statefulpartitionedcall_args_2
identity??layer1/StatefulPartitionedCall?layer2/StatefulPartitionedCall?layer3/StatefulPartitionedCall?)mf_embedding_item/StatefulPartitionedCall?)mf_embedding_user/StatefulPartitionedCall?*mlp_embedding_item/StatefulPartitionedCall?*mlp_embedding_user/StatefulPartitionedCall?"prediction/StatefulPartitionedCall?
*mlp_embedding_item/StatefulPartitionedCallStatefulPartitionedCall
item_input1mlp_embedding_item_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:????????? *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*W
fRRP
N__inference_mlp_embedding_item_layer_call_and_return_conditional_losses_8758482,
*mlp_embedding_item/StatefulPartitionedCall?
*mlp_embedding_user/StatefulPartitionedCallStatefulPartitionedCall
user_input1mlp_embedding_user_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:????????? *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*W
fRRP
N__inference_mlp_embedding_user_layer_call_and_return_conditional_losses_8758682,
*mlp_embedding_user/StatefulPartitionedCall?
flatten_2/PartitionedCallPartitionedCall3mlp_embedding_user/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:????????? *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_8758842
flatten_2/PartitionedCall?
flatten_3/PartitionedCallPartitionedCall3mlp_embedding_item/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:????????? *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_8758982
flatten_3/PartitionedCall?
concatenate/PartitionedCallPartitionedCall"flatten_2/PartitionedCall:output:0"flatten_3/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????@*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_8759132
concatenate/PartitionedCall?
layer1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0%layer1_statefulpartitionedcall_args_1%layer1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:????????? *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*K
fFRD
B__inference_layer1_layer_call_and_return_conditional_losses_8759342 
layer1/StatefulPartitionedCall?
)mf_embedding_item/StatefulPartitionedCallStatefulPartitionedCall
item_input0mf_embedding_item_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*V
fQRO
M__inference_mf_embedding_item_layer_call_and_return_conditional_losses_8759562+
)mf_embedding_item/StatefulPartitionedCall?
)mf_embedding_user/StatefulPartitionedCallStatefulPartitionedCall
user_input0mf_embedding_user_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*V
fQRO
M__inference_mf_embedding_user_layer_call_and_return_conditional_losses_8759762+
)mf_embedding_user/StatefulPartitionedCall?
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0%layer2_statefulpartitionedcall_args_1%layer2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*K
fFRD
B__inference_layer2_layer_call_and_return_conditional_losses_8759982 
layer2/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall2mf_embedding_user/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_8760162
flatten/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall2mf_embedding_item/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_8760302
flatten_1/PartitionedCall?
multiply/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*M
fHRF
D__inference_multiply_layer_call_and_return_conditional_losses_8760442
multiply/PartitionedCall?
layer3/StatefulPartitionedCallStatefulPartitionedCall'layer2/StatefulPartitionedCall:output:0%layer3_statefulpartitionedcall_args_1%layer3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*K
fFRD
B__inference_layer3_layer_call_and_return_conditional_losses_8760652 
layer3/StatefulPartitionedCall?
concatenate_1/PartitionedCallPartitionedCall!multiply/PartitionedCall:output:0'layer3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_8760842
concatenate_1/PartitionedCall?
"prediction/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0)prediction_statefulpartitionedcall_args_1)prediction_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*O
fJRH
F__inference_prediction_layer_call_and_return_conditional_losses_8761042$
"prediction/StatefulPartitionedCall?
/mlp_embedding_user/embeddings/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    21
/mlp_embedding_user/embeddings/Regularizer/Const?
/mlp_embedding_item/embeddings/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    21
/mlp_embedding_item/embeddings/Regularizer/Const?
.mf_embedding_user/embeddings/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    20
.mf_embedding_user/embeddings/Regularizer/Const?
.mf_embedding_item/embeddings/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    20
.mf_embedding_item/embeddings/Regularizer/Const?
layer1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
layer1/kernel/Regularizer/Const?
layer2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
layer2/kernel/Regularizer/Const?
layer3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
layer3/kernel/Regularizer/Const?
IdentityIdentity+prediction/StatefulPartitionedCall:output:0^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall^layer3/StatefulPartitionedCall*^mf_embedding_item/StatefulPartitionedCall*^mf_embedding_user/StatefulPartitionedCall+^mlp_embedding_item/StatefulPartitionedCall+^mlp_embedding_user/StatefulPartitionedCall#^prediction/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:?????????:?????????::::::::::::2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2@
layer3/StatefulPartitionedCalllayer3/StatefulPartitionedCall2V
)mf_embedding_item/StatefulPartitionedCall)mf_embedding_item/StatefulPartitionedCall2V
)mf_embedding_user/StatefulPartitionedCall)mf_embedding_user/StatefulPartitionedCall2X
*mlp_embedding_item/StatefulPartitionedCall*mlp_embedding_item/StatefulPartitionedCall2X
*mlp_embedding_user/StatefulPartitionedCall*mlp_embedding_user/StatefulPartitionedCall2H
"prediction/StatefulPartitionedCall"prediction/StatefulPartitionedCall:* &
$
_user_specified_name
user_input:*&
$
_user_specified_name
item_input
?
u
I__inference_concatenate_1_layer_call_and_return_conditional_losses_876698
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?
,
__inference_loss_fn_5_876752
identity?
layer2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
layer2/kernel/Regularizer/Constk
IdentityIdentity(layer2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?
?
+__inference_prediction_layer_call_fn_876722

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*O
fJRH
F__inference_prediction_layer_call_and_return_conditional_losses_8761042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
3__inference_mlp_embedding_user_layer_call_fn_876511

inputs"
statefulpartitionedcall_args_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:????????? *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*W
fRRP
N__inference_mlp_embedding_user_layer_call_and_return_conditional_losses_8758682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?\
?
A__inference_model_layer_call_and_return_conditional_losses_876458
inputs_0
inputs_1.
*mlp_embedding_item_embedding_lookup_876389.
*mlp_embedding_user_embedding_lookup_876394)
%layer1_matmul_readvariableop_resource*
&layer1_biasadd_readvariableop_resource-
)mf_embedding_item_embedding_lookup_876412-
)mf_embedding_user_embedding_lookup_876417)
%layer2_matmul_readvariableop_resource*
&layer2_biasadd_readvariableop_resource)
%layer3_matmul_readvariableop_resource*
&layer3_biasadd_readvariableop_resource-
)prediction_matmul_readvariableop_resource.
*prediction_biasadd_readvariableop_resource
identity??layer1/BiasAdd/ReadVariableOp?layer1/MatMul/ReadVariableOp?layer2/BiasAdd/ReadVariableOp?layer2/MatMul/ReadVariableOp?layer3/BiasAdd/ReadVariableOp?layer3/MatMul/ReadVariableOp?"mf_embedding_item/embedding_lookup?"mf_embedding_user/embedding_lookup?#mlp_embedding_item/embedding_lookup?#mlp_embedding_user/embedding_lookup?!prediction/BiasAdd/ReadVariableOp? prediction/MatMul/ReadVariableOp?
#mlp_embedding_item/embedding_lookupResourceGather*mlp_embedding_item_embedding_lookup_876389inputs_1*
Tindices0*=
_class3
1/loc:@mlp_embedding_item/embedding_lookup/876389*+
_output_shapes
:????????? *
dtype02%
#mlp_embedding_item/embedding_lookup?
,mlp_embedding_item/embedding_lookup/IdentityIdentity,mlp_embedding_item/embedding_lookup:output:0*
T0*=
_class3
1/loc:@mlp_embedding_item/embedding_lookup/876389*+
_output_shapes
:????????? 2.
,mlp_embedding_item/embedding_lookup/Identity?
.mlp_embedding_item/embedding_lookup/Identity_1Identity5mlp_embedding_item/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:????????? 20
.mlp_embedding_item/embedding_lookup/Identity_1?
#mlp_embedding_user/embedding_lookupResourceGather*mlp_embedding_user_embedding_lookup_876394inputs_0*
Tindices0*=
_class3
1/loc:@mlp_embedding_user/embedding_lookup/876394*+
_output_shapes
:????????? *
dtype02%
#mlp_embedding_user/embedding_lookup?
,mlp_embedding_user/embedding_lookup/IdentityIdentity,mlp_embedding_user/embedding_lookup:output:0*
T0*=
_class3
1/loc:@mlp_embedding_user/embedding_lookup/876394*+
_output_shapes
:????????? 2.
,mlp_embedding_user/embedding_lookup/Identity?
.mlp_embedding_user/embedding_lookup/Identity_1Identity5mlp_embedding_user/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:????????? 20
.mlp_embedding_user/embedding_lookup/Identity_1s
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"????    2
flatten_2/Const?
flatten_2/ReshapeReshape7mlp_embedding_user/embedding_lookup/Identity_1:output:0flatten_2/Const:output:0*
T0*'
_output_shapes
:????????? 2
flatten_2/Reshapes
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"????    2
flatten_3/Const?
flatten_3/ReshapeReshape7mlp_embedding_item/embedding_lookup/Identity_1:output:0flatten_3/Const:output:0*
T0*'
_output_shapes
:????????? 2
flatten_3/Reshapet
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2flatten_2/Reshape:output:0flatten_3/Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????@2
concatenate/concat?
layer1/MatMul/ReadVariableOpReadVariableOp%layer1_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
layer1/MatMul/ReadVariableOp?
layer1/MatMulMatMulconcatenate/concat:output:0$layer1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
layer1/MatMul?
layer1/BiasAdd/ReadVariableOpReadVariableOp&layer1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
layer1/BiasAdd/ReadVariableOp?
layer1/BiasAddBiasAddlayer1/MatMul:product:0%layer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
layer1/BiasAddm
layer1/ReluRelulayer1/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
layer1/Relu?
"mf_embedding_item/embedding_lookupResourceGather)mf_embedding_item_embedding_lookup_876412inputs_1*
Tindices0*<
_class2
0.loc:@mf_embedding_item/embedding_lookup/876412*+
_output_shapes
:?????????*
dtype02$
"mf_embedding_item/embedding_lookup?
+mf_embedding_item/embedding_lookup/IdentityIdentity+mf_embedding_item/embedding_lookup:output:0*
T0*<
_class2
0.loc:@mf_embedding_item/embedding_lookup/876412*+
_output_shapes
:?????????2-
+mf_embedding_item/embedding_lookup/Identity?
-mf_embedding_item/embedding_lookup/Identity_1Identity4mf_embedding_item/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2/
-mf_embedding_item/embedding_lookup/Identity_1?
"mf_embedding_user/embedding_lookupResourceGather)mf_embedding_user_embedding_lookup_876417inputs_0*
Tindices0*<
_class2
0.loc:@mf_embedding_user/embedding_lookup/876417*+
_output_shapes
:?????????*
dtype02$
"mf_embedding_user/embedding_lookup?
+mf_embedding_user/embedding_lookup/IdentityIdentity+mf_embedding_user/embedding_lookup:output:0*
T0*<
_class2
0.loc:@mf_embedding_user/embedding_lookup/876417*+
_output_shapes
:?????????2-
+mf_embedding_user/embedding_lookup/Identity?
-mf_embedding_user/embedding_lookup/Identity_1Identity4mf_embedding_user/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2/
-mf_embedding_user/embedding_lookup/Identity_1?
layer2/MatMul/ReadVariableOpReadVariableOp%layer2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
layer2/MatMul/ReadVariableOp?
layer2/MatMulMatMullayer1/Relu:activations:0$layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
layer2/MatMul?
layer2/BiasAdd/ReadVariableOpReadVariableOp&layer2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
layer2/BiasAdd/ReadVariableOp?
layer2/BiasAddBiasAddlayer2/MatMul:product:0%layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
layer2/BiasAddm
layer2/ReluRelulayer2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
layer2/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Const?
flatten/ReshapeReshape6mf_embedding_user/embedding_lookup/Identity_1:output:0flatten/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten/Reshapes
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_1/Const?
flatten_1/ReshapeReshape6mf_embedding_item/embedding_lookup/Identity_1:output:0flatten_1/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_1/Reshape?
multiply/mulMulflatten/Reshape:output:0flatten_1/Reshape:output:0*
T0*'
_output_shapes
:?????????2
multiply/mul?
layer3/MatMul/ReadVariableOpReadVariableOp%layer3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
layer3/MatMul/ReadVariableOp?
layer3/MatMulMatMullayer2/Relu:activations:0$layer3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
layer3/MatMul?
layer3/BiasAdd/ReadVariableOpReadVariableOp&layer3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
layer3/BiasAdd/ReadVariableOp?
layer3/BiasAddBiasAddlayer3/MatMul:product:0%layer3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
layer3/BiasAddm
layer3/ReluRelulayer3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
layer3/Relux
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2multiply/mul:z:0layer3/Relu:activations:0"concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatenate_1/concat?
 prediction/MatMul/ReadVariableOpReadVariableOp)prediction_matmul_readvariableop_resource*
_output_shapes

:*
dtype02"
 prediction/MatMul/ReadVariableOp?
prediction/MatMulMatMulconcatenate_1/concat:output:0(prediction/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
prediction/MatMul?
!prediction/BiasAdd/ReadVariableOpReadVariableOp*prediction_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!prediction/BiasAdd/ReadVariableOp?
prediction/BiasAddBiasAddprediction/MatMul:product:0)prediction/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
prediction/BiasAdd?
prediction/SigmoidSigmoidprediction/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
prediction/Sigmoid?
/mlp_embedding_user/embeddings/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    21
/mlp_embedding_user/embeddings/Regularizer/Const?
/mlp_embedding_item/embeddings/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    21
/mlp_embedding_item/embeddings/Regularizer/Const?
.mf_embedding_user/embeddings/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    20
.mf_embedding_user/embeddings/Regularizer/Const?
.mf_embedding_item/embeddings/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    20
.mf_embedding_item/embeddings/Regularizer/Const?
layer1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
layer1/kernel/Regularizer/Const?
layer2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
layer2/kernel/Regularizer/Const?
layer3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
layer3/kernel/Regularizer/Const?
IdentityIdentityprediction/Sigmoid:y:0^layer1/BiasAdd/ReadVariableOp^layer1/MatMul/ReadVariableOp^layer2/BiasAdd/ReadVariableOp^layer2/MatMul/ReadVariableOp^layer3/BiasAdd/ReadVariableOp^layer3/MatMul/ReadVariableOp#^mf_embedding_item/embedding_lookup#^mf_embedding_user/embedding_lookup$^mlp_embedding_item/embedding_lookup$^mlp_embedding_user/embedding_lookup"^prediction/BiasAdd/ReadVariableOp!^prediction/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:?????????:?????????::::::::::::2>
layer1/BiasAdd/ReadVariableOplayer1/BiasAdd/ReadVariableOp2<
layer1/MatMul/ReadVariableOplayer1/MatMul/ReadVariableOp2>
layer2/BiasAdd/ReadVariableOplayer2/BiasAdd/ReadVariableOp2<
layer2/MatMul/ReadVariableOplayer2/MatMul/ReadVariableOp2>
layer3/BiasAdd/ReadVariableOplayer3/BiasAdd/ReadVariableOp2<
layer3/MatMul/ReadVariableOplayer3/MatMul/ReadVariableOp2H
"mf_embedding_item/embedding_lookup"mf_embedding_item/embedding_lookup2H
"mf_embedding_user/embedding_lookup"mf_embedding_user/embedding_lookup2J
#mlp_embedding_item/embedding_lookup#mlp_embedding_item/embedding_lookup2J
#mlp_embedding_user/embedding_lookup#mlp_embedding_user/embedding_lookup2F
!prediction/BiasAdd/ReadVariableOp!prediction/BiasAdd/ReadVariableOp2D
 prediction/MatMul/ReadVariableOp prediction/MatMul/ReadVariableOp:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?^
?	
!__inference__wrapped_model_875833

user_input

item_input4
0model_mlp_embedding_item_embedding_lookup_8757714
0model_mlp_embedding_user_embedding_lookup_875776/
+model_layer1_matmul_readvariableop_resource0
,model_layer1_biasadd_readvariableop_resource3
/model_mf_embedding_item_embedding_lookup_8757943
/model_mf_embedding_user_embedding_lookup_875799/
+model_layer2_matmul_readvariableop_resource0
,model_layer2_biasadd_readvariableop_resource/
+model_layer3_matmul_readvariableop_resource0
,model_layer3_biasadd_readvariableop_resource3
/model_prediction_matmul_readvariableop_resource4
0model_prediction_biasadd_readvariableop_resource
identity??#model/layer1/BiasAdd/ReadVariableOp?"model/layer1/MatMul/ReadVariableOp?#model/layer2/BiasAdd/ReadVariableOp?"model/layer2/MatMul/ReadVariableOp?#model/layer3/BiasAdd/ReadVariableOp?"model/layer3/MatMul/ReadVariableOp?(model/mf_embedding_item/embedding_lookup?(model/mf_embedding_user/embedding_lookup?)model/mlp_embedding_item/embedding_lookup?)model/mlp_embedding_user/embedding_lookup?'model/prediction/BiasAdd/ReadVariableOp?&model/prediction/MatMul/ReadVariableOp?
)model/mlp_embedding_item/embedding_lookupResourceGather0model_mlp_embedding_item_embedding_lookup_875771
item_input*
Tindices0*C
_class9
75loc:@model/mlp_embedding_item/embedding_lookup/875771*+
_output_shapes
:????????? *
dtype02+
)model/mlp_embedding_item/embedding_lookup?
2model/mlp_embedding_item/embedding_lookup/IdentityIdentity2model/mlp_embedding_item/embedding_lookup:output:0*
T0*C
_class9
75loc:@model/mlp_embedding_item/embedding_lookup/875771*+
_output_shapes
:????????? 24
2model/mlp_embedding_item/embedding_lookup/Identity?
4model/mlp_embedding_item/embedding_lookup/Identity_1Identity;model/mlp_embedding_item/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:????????? 26
4model/mlp_embedding_item/embedding_lookup/Identity_1?
)model/mlp_embedding_user/embedding_lookupResourceGather0model_mlp_embedding_user_embedding_lookup_875776
user_input*
Tindices0*C
_class9
75loc:@model/mlp_embedding_user/embedding_lookup/875776*+
_output_shapes
:????????? *
dtype02+
)model/mlp_embedding_user/embedding_lookup?
2model/mlp_embedding_user/embedding_lookup/IdentityIdentity2model/mlp_embedding_user/embedding_lookup:output:0*
T0*C
_class9
75loc:@model/mlp_embedding_user/embedding_lookup/875776*+
_output_shapes
:????????? 24
2model/mlp_embedding_user/embedding_lookup/Identity?
4model/mlp_embedding_user/embedding_lookup/Identity_1Identity;model/mlp_embedding_user/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:????????? 26
4model/mlp_embedding_user/embedding_lookup/Identity_1
model/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"????    2
model/flatten_2/Const?
model/flatten_2/ReshapeReshape=model/mlp_embedding_user/embedding_lookup/Identity_1:output:0model/flatten_2/Const:output:0*
T0*'
_output_shapes
:????????? 2
model/flatten_2/Reshape
model/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"????    2
model/flatten_3/Const?
model/flatten_3/ReshapeReshape=model/mlp_embedding_item/embedding_lookup/Identity_1:output:0model/flatten_3/Const:output:0*
T0*'
_output_shapes
:????????? 2
model/flatten_3/Reshape?
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model/concatenate/concat/axis?
model/concatenate/concatConcatV2 model/flatten_2/Reshape:output:0 model/flatten_3/Reshape:output:0&model/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????@2
model/concatenate/concat?
"model/layer1/MatMul/ReadVariableOpReadVariableOp+model_layer1_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02$
"model/layer1/MatMul/ReadVariableOp?
model/layer1/MatMulMatMul!model/concatenate/concat:output:0*model/layer1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
model/layer1/MatMul?
#model/layer1/BiasAdd/ReadVariableOpReadVariableOp,model_layer1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#model/layer1/BiasAdd/ReadVariableOp?
model/layer1/BiasAddBiasAddmodel/layer1/MatMul:product:0+model/layer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
model/layer1/BiasAdd
model/layer1/ReluRelumodel/layer1/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
model/layer1/Relu?
(model/mf_embedding_item/embedding_lookupResourceGather/model_mf_embedding_item_embedding_lookup_875794
item_input*
Tindices0*B
_class8
64loc:@model/mf_embedding_item/embedding_lookup/875794*+
_output_shapes
:?????????*
dtype02*
(model/mf_embedding_item/embedding_lookup?
1model/mf_embedding_item/embedding_lookup/IdentityIdentity1model/mf_embedding_item/embedding_lookup:output:0*
T0*B
_class8
64loc:@model/mf_embedding_item/embedding_lookup/875794*+
_output_shapes
:?????????23
1model/mf_embedding_item/embedding_lookup/Identity?
3model/mf_embedding_item/embedding_lookup/Identity_1Identity:model/mf_embedding_item/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????25
3model/mf_embedding_item/embedding_lookup/Identity_1?
(model/mf_embedding_user/embedding_lookupResourceGather/model_mf_embedding_user_embedding_lookup_875799
user_input*
Tindices0*B
_class8
64loc:@model/mf_embedding_user/embedding_lookup/875799*+
_output_shapes
:?????????*
dtype02*
(model/mf_embedding_user/embedding_lookup?
1model/mf_embedding_user/embedding_lookup/IdentityIdentity1model/mf_embedding_user/embedding_lookup:output:0*
T0*B
_class8
64loc:@model/mf_embedding_user/embedding_lookup/875799*+
_output_shapes
:?????????23
1model/mf_embedding_user/embedding_lookup/Identity?
3model/mf_embedding_user/embedding_lookup/Identity_1Identity:model/mf_embedding_user/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????25
3model/mf_embedding_user/embedding_lookup/Identity_1?
"model/layer2/MatMul/ReadVariableOpReadVariableOp+model_layer2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02$
"model/layer2/MatMul/ReadVariableOp?
model/layer2/MatMulMatMulmodel/layer1/Relu:activations:0*model/layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/layer2/MatMul?
#model/layer2/BiasAdd/ReadVariableOpReadVariableOp,model_layer2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#model/layer2/BiasAdd/ReadVariableOp?
model/layer2/BiasAddBiasAddmodel/layer2/MatMul:product:0+model/layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/layer2/BiasAdd
model/layer2/ReluRelumodel/layer2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model/layer2/Relu{
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model/flatten/Const?
model/flatten/ReshapeReshape<model/mf_embedding_user/embedding_lookup/Identity_1:output:0model/flatten/Const:output:0*
T0*'
_output_shapes
:?????????2
model/flatten/Reshape
model/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model/flatten_1/Const?
model/flatten_1/ReshapeReshape<model/mf_embedding_item/embedding_lookup/Identity_1:output:0model/flatten_1/Const:output:0*
T0*'
_output_shapes
:?????????2
model/flatten_1/Reshape?
model/multiply/mulMulmodel/flatten/Reshape:output:0 model/flatten_1/Reshape:output:0*
T0*'
_output_shapes
:?????????2
model/multiply/mul?
"model/layer3/MatMul/ReadVariableOpReadVariableOp+model_layer3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02$
"model/layer3/MatMul/ReadVariableOp?
model/layer3/MatMulMatMulmodel/layer2/Relu:activations:0*model/layer3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/layer3/MatMul?
#model/layer3/BiasAdd/ReadVariableOpReadVariableOp,model_layer3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#model/layer3/BiasAdd/ReadVariableOp?
model/layer3/BiasAddBiasAddmodel/layer3/MatMul:product:0+model/layer3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/layer3/BiasAdd
model/layer3/ReluRelumodel/layer3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model/layer3/Relu?
model/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2!
model/concatenate_1/concat/axis?
model/concatenate_1/concatConcatV2model/multiply/mul:z:0model/layer3/Relu:activations:0(model/concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
model/concatenate_1/concat?
&model/prediction/MatMul/ReadVariableOpReadVariableOp/model_prediction_matmul_readvariableop_resource*
_output_shapes

:*
dtype02(
&model/prediction/MatMul/ReadVariableOp?
model/prediction/MatMulMatMul#model/concatenate_1/concat:output:0.model/prediction/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/prediction/MatMul?
'model/prediction/BiasAdd/ReadVariableOpReadVariableOp0model_prediction_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model/prediction/BiasAdd/ReadVariableOp?
model/prediction/BiasAddBiasAdd!model/prediction/MatMul:product:0/model/prediction/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/prediction/BiasAdd?
model/prediction/SigmoidSigmoid!model/prediction/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model/prediction/Sigmoid?
IdentityIdentitymodel/prediction/Sigmoid:y:0$^model/layer1/BiasAdd/ReadVariableOp#^model/layer1/MatMul/ReadVariableOp$^model/layer2/BiasAdd/ReadVariableOp#^model/layer2/MatMul/ReadVariableOp$^model/layer3/BiasAdd/ReadVariableOp#^model/layer3/MatMul/ReadVariableOp)^model/mf_embedding_item/embedding_lookup)^model/mf_embedding_user/embedding_lookup*^model/mlp_embedding_item/embedding_lookup*^model/mlp_embedding_user/embedding_lookup(^model/prediction/BiasAdd/ReadVariableOp'^model/prediction/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:?????????:?????????::::::::::::2J
#model/layer1/BiasAdd/ReadVariableOp#model/layer1/BiasAdd/ReadVariableOp2H
"model/layer1/MatMul/ReadVariableOp"model/layer1/MatMul/ReadVariableOp2J
#model/layer2/BiasAdd/ReadVariableOp#model/layer2/BiasAdd/ReadVariableOp2H
"model/layer2/MatMul/ReadVariableOp"model/layer2/MatMul/ReadVariableOp2J
#model/layer3/BiasAdd/ReadVariableOp#model/layer3/BiasAdd/ReadVariableOp2H
"model/layer3/MatMul/ReadVariableOp"model/layer3/MatMul/ReadVariableOp2T
(model/mf_embedding_item/embedding_lookup(model/mf_embedding_item/embedding_lookup2T
(model/mf_embedding_user/embedding_lookup(model/mf_embedding_user/embedding_lookup2V
)model/mlp_embedding_item/embedding_lookup)model/mlp_embedding_item/embedding_lookup2V
)model/mlp_embedding_user/embedding_lookup)model/mlp_embedding_user/embedding_lookup2R
'model/prediction/BiasAdd/ReadVariableOp'model/prediction/BiasAdd/ReadVariableOp2P
&model/prediction/MatMul/ReadVariableOp&model/prediction/MatMul/ReadVariableOp:* &
$
_user_specified_name
user_input:*&
$
_user_specified_name
item_input
?
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_875898

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????    2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:????????? 2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0**
_input_shapes
:????????? :& "
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
A

item_input3
serving_default_item_input:0?????????
A

user_input3
serving_default_user_input:0?????????>

prediction0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?m
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer_with_weights-3
	layer-8

layer_with_weights-4

layer-9
layer-10
layer-11
layer_with_weights-5
layer-12
layer-13
layer_with_weights-6
layer-14
layer-15
layer_with_weights-7
layer-16
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"?h
_tf_keras_model?h{"class_name": "Model", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 1], "dtype": "int32", "sparse": false, "ragged": false, "name": "user_input"}, "name": "user_input", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": [null, 1], "dtype": "int32", "sparse": false, "ragged": false, "name": "item_input"}, "name": "item_input", "inbound_nodes": []}, {"class_name": "Embedding", "config": {"name": "mlp_embedding_user", "trainable": true, "batch_input_shape": [null, 1], "dtype": "float32", "input_dim": 7040, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.01, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "mlp_embedding_user", "inbound_nodes": [[["user_input", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "mlp_embedding_item", "trainable": true, "batch_input_shape": [null, 1], "dtype": "float32", "input_dim": 3706, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.01, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "mlp_embedding_item", "inbound_nodes": [[["item_input", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_2", "inbound_nodes": [[["mlp_embedding_user", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_3", "inbound_nodes": [[["mlp_embedding_item", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["flatten_2", 0, 0, {}], ["flatten_3", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "mf_embedding_user", "trainable": true, "batch_input_shape": [null, 1], "dtype": "float32", "input_dim": 7040, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.01, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "mf_embedding_user", "inbound_nodes": [[["user_input", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "mf_embedding_item", "trainable": true, "batch_input_shape": [null, 1], "dtype": "float32", "input_dim": 3706, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.01, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "mf_embedding_item", "inbound_nodes": [[["item_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "layer1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer1", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["mf_embedding_user", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["mf_embedding_item", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "layer2", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer2", "inbound_nodes": [[["layer1", 0, 0, {}]]]}, {"class_name": "Multiply", "config": {"name": "multiply", "trainable": true, "dtype": "float32"}, "name": "multiply", "inbound_nodes": [[["flatten", 0, 0, {}], ["flatten_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "layer3", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer3", "inbound_nodes": [[["layer2", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_1", "inbound_nodes": [[["multiply", 0, 0, {}], ["layer3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "prediction", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "prediction", "inbound_nodes": [[["concatenate_1", 0, 0, {}]]]}], "input_layers": [["user_input", 0, 0], ["item_input", 0, 0]], "output_layers": [["prediction", 0, 0]]}, "input_spec": [null, null], "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 1], "dtype": "int32", "sparse": false, "ragged": false, "name": "user_input"}, "name": "user_input", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": [null, 1], "dtype": "int32", "sparse": false, "ragged": false, "name": "item_input"}, "name": "item_input", "inbound_nodes": []}, {"class_name": "Embedding", "config": {"name": "mlp_embedding_user", "trainable": true, "batch_input_shape": [null, 1], "dtype": "float32", "input_dim": 7040, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.01, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "mlp_embedding_user", "inbound_nodes": [[["user_input", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "mlp_embedding_item", "trainable": true, "batch_input_shape": [null, 1], "dtype": "float32", "input_dim": 3706, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.01, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "mlp_embedding_item", "inbound_nodes": [[["item_input", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_2", "inbound_nodes": [[["mlp_embedding_user", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_3", "inbound_nodes": [[["mlp_embedding_item", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["flatten_2", 0, 0, {}], ["flatten_3", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "mf_embedding_user", "trainable": true, "batch_input_shape": [null, 1], "dtype": "float32", "input_dim": 7040, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.01, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "mf_embedding_user", "inbound_nodes": [[["user_input", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "mf_embedding_item", "trainable": true, "batch_input_shape": [null, 1], "dtype": "float32", "input_dim": 3706, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.01, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "mf_embedding_item", "inbound_nodes": [[["item_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "layer1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer1", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["mf_embedding_user", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["mf_embedding_item", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "layer2", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer2", "inbound_nodes": [[["layer1", 0, 0, {}]]]}, {"class_name": "Multiply", "config": {"name": "multiply", "trainable": true, "dtype": "float32"}, "name": "multiply", "inbound_nodes": [[["flatten", 0, 0, {}], ["flatten_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "layer3", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer3", "inbound_nodes": [[["layer2", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_1", "inbound_nodes": [[["multiply", 0, 0, {}], ["layer3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "prediction", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "prediction", "inbound_nodes": [[["concatenate_1", 0, 0, {}]]]}], "input_layers": [["user_input", 0, 0], ["item_input", 0, 0]], "output_layers": [["prediction", 0, 0]]}}, "training_config": {"loss": "binary_crossentropy", "metrics": ["acc"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0005000000237487257, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "user_input", "dtype": "int32", "sparse": false, "ragged": false, "batch_input_shape": [null, 1], "config": {"batch_input_shape": [null, 1], "dtype": "int32", "sparse": false, "ragged": false, "name": "user_input"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "item_input", "dtype": "int32", "sparse": false, "ragged": false, "batch_input_shape": [null, 1], "config": {"batch_input_shape": [null, 1], "dtype": "int32", "sparse": false, "ragged": false, "name": "item_input"}}
?

embeddings
regularization_losses
	variables
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Embedding", "name": "mlp_embedding_user", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 1], "config": {"name": "mlp_embedding_user", "trainable": true, "batch_input_shape": [null, 1], "dtype": "float32", "input_dim": 7040, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.01, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}}
?

embeddings
regularization_losses
	variables
 trainable_variables
!	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Embedding", "name": "mlp_embedding_item", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 1], "config": {"name": "mlp_embedding_item", "trainable": true, "batch_input_shape": [null, 1], "dtype": "float32", "input_dim": 3706, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.01, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}}
?
"regularization_losses
#	variables
$trainable_variables
%	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
&regularization_losses
'	variables
(trainable_variables
)	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
*regularization_losses
+	variables
,trainable_variables
-	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}}
?
.
embeddings
/regularization_losses
0	variables
1trainable_variables
2	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Embedding", "name": "mf_embedding_user", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 1], "config": {"name": "mf_embedding_user", "trainable": true, "batch_input_shape": [null, 1], "dtype": "float32", "input_dim": 7040, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.01, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}}
?
3
embeddings
4regularization_losses
5	variables
6trainable_variables
7	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Embedding", "name": "mf_embedding_item", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 1], "config": {"name": "mf_embedding_item", "trainable": true, "batch_input_shape": [null, 1], "dtype": "float32", "input_dim": 3706, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.01, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}}
?

8kernel
9bias
:regularization_losses
;	variables
<trainable_variables
=	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "layer1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "layer1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}}
?
>regularization_losses
?	variables
@trainable_variables
A	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

Fkernel
Gbias
Hregularization_losses
I	variables
Jtrainable_variables
K	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "layer2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "layer2", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}}
?
Lregularization_losses
M	variables
Ntrainable_variables
O	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Multiply", "name": "multiply", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "multiply", "trainable": true, "dtype": "float32"}}
?

Pkernel
Qbias
Rregularization_losses
S	variables
Ttrainable_variables
U	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "layer3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "layer3", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}}
?
Vregularization_losses
W	variables
Xtrainable_variables
Y	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}}
?

Zkernel
[bias
\regularization_losses
]	variables
^trainable_variables
_	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "prediction", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "prediction", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}}
?
`iter

abeta_1

bbeta_2
	cdecay
dlearning_ratem?m?.m?3m?8m?9m?Fm?Gm?Pm?Qm?Zm?[m?v?v?.v?3v?8v?9v?Fv?Gv?Pv?Qv?Zv?[v?"
	optimizer
X
?0
?1
?2
?3
?4
?5
?6"
trackable_list_wrapper
v
0
1
.2
33
84
95
F6
G7
P8
Q9
Z10
[11"
trackable_list_wrapper
v
0
1
.2
33
84
95
F6
G7
P8
Q9
Z10
[11"
trackable_list_wrapper
?
emetrics
regularization_losses

flayers
	variables
glayer_regularization_losses
hnon_trainable_variables
trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
0:.	?7 2mlp_embedding_user/embeddings
(
?0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
?
imetrics
regularization_losses

jlayers
	variables
klayer_regularization_losses
lnon_trainable_variables
trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0:.	? 2mlp_embedding_item/embeddings
(
?0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
?
mmetrics
regularization_losses

nlayers
	variables
olayer_regularization_losses
pnon_trainable_variables
 trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
qmetrics
"regularization_losses

rlayers
#	variables
slayer_regularization_losses
tnon_trainable_variables
$trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
umetrics
&regularization_losses

vlayers
'	variables
wlayer_regularization_losses
xnon_trainable_variables
(trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
ymetrics
*regularization_losses

zlayers
+	variables
{layer_regularization_losses
|non_trainable_variables
,trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-	?72mf_embedding_user/embeddings
(
?0"
trackable_list_wrapper
'
.0"
trackable_list_wrapper
'
.0"
trackable_list_wrapper
?
}metrics
/regularization_losses

~layers
0	variables
layer_regularization_losses
?non_trainable_variables
1trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-	?2mf_embedding_item/embeddings
(
?0"
trackable_list_wrapper
'
30"
trackable_list_wrapper
'
30"
trackable_list_wrapper
?
?metrics
4regularization_losses
?layers
5	variables
 ?layer_regularization_losses
?non_trainable_variables
6trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:@ 2layer1/kernel
: 2layer1/bias
(
?0"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
?
?metrics
:regularization_losses
?layers
;	variables
 ?layer_regularization_losses
?non_trainable_variables
<trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
>regularization_losses
?layers
?	variables
 ?layer_regularization_losses
?non_trainable_variables
@trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
Bregularization_losses
?layers
C	variables
 ?layer_regularization_losses
?non_trainable_variables
Dtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
: 2layer2/kernel
:2layer2/bias
(
?0"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
?
?metrics
Hregularization_losses
?layers
I	variables
 ?layer_regularization_losses
?non_trainable_variables
Jtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
Lregularization_losses
?layers
M	variables
 ?layer_regularization_losses
?non_trainable_variables
Ntrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:2layer3/kernel
:2layer3/bias
(
?0"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
?
?metrics
Rregularization_losses
?layers
S	variables
 ?layer_regularization_losses
?non_trainable_variables
Ttrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
Vregularization_losses
?layers
W	variables
 ?layer_regularization_losses
?non_trainable_variables
Xtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!2prediction/kernel
:2prediction/bias
 "
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
?
?metrics
\regularization_losses
?layers
]	variables
 ?layer_regularization_losses
?non_trainable_variables
^trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
(
?0"
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?
_fn_kwargs
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MeanMetricWrapper", "name": "acc", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "acc", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?regularization_losses
?layers
?	variables
 ?layer_regularization_losses
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
5:3	?7 2$Adam/mlp_embedding_user/embeddings/m
5:3	? 2$Adam/mlp_embedding_item/embeddings/m
4:2	?72#Adam/mf_embedding_user/embeddings/m
4:2	?2#Adam/mf_embedding_item/embeddings/m
$:"@ 2Adam/layer1/kernel/m
: 2Adam/layer1/bias/m
$:" 2Adam/layer2/kernel/m
:2Adam/layer2/bias/m
$:"2Adam/layer3/kernel/m
:2Adam/layer3/bias/m
(:&2Adam/prediction/kernel/m
": 2Adam/prediction/bias/m
5:3	?7 2$Adam/mlp_embedding_user/embeddings/v
5:3	? 2$Adam/mlp_embedding_item/embeddings/v
4:2	?72#Adam/mf_embedding_user/embeddings/v
4:2	?2#Adam/mf_embedding_item/embeddings/v
$:"@ 2Adam/layer1/kernel/v
: 2Adam/layer1/bias/v
$:" 2Adam/layer2/kernel/v
:2Adam/layer2/bias/v
$:"2Adam/layer3/kernel/v
:2Adam/layer3/bias/v
(:&2Adam/prediction/kernel/v
": 2Adam/prediction/bias/v
?2?
&__inference_model_layer_call_fn_876278
&__inference_model_layer_call_fn_876221
&__inference_model_layer_call_fn_876476
&__inference_model_layer_call_fn_876494?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
!__inference__wrapped_model_875833?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *T?Q
O?L
$?!

user_input?????????
$?!

item_input?????????
?2?
A__inference_model_layer_call_and_return_conditional_losses_876385
A__inference_model_layer_call_and_return_conditional_losses_876163
A__inference_model_layer_call_and_return_conditional_losses_876124
A__inference_model_layer_call_and_return_conditional_losses_876458?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
3__inference_mlp_embedding_user_layer_call_fn_876511?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
N__inference_mlp_embedding_user_layer_call_and_return_conditional_losses_876505?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
3__inference_mlp_embedding_item_layer_call_fn_876528?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
N__inference_mlp_embedding_item_layer_call_and_return_conditional_losses_876522?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_flatten_2_layer_call_fn_876539?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_flatten_2_layer_call_and_return_conditional_losses_876534?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_flatten_3_layer_call_fn_876550?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_flatten_3_layer_call_and_return_conditional_losses_876545?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_concatenate_layer_call_fn_876563?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_concatenate_layer_call_and_return_conditional_losses_876557?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
2__inference_mf_embedding_user_layer_call_fn_876580?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_mf_embedding_user_layer_call_and_return_conditional_losses_876574?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
2__inference_mf_embedding_item_layer_call_fn_876597?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_mf_embedding_item_layer_call_and_return_conditional_losses_876591?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_layer1_layer_call_fn_876617?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_layer1_layer_call_and_return_conditional_losses_876610?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_flatten_layer_call_fn_876628?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_flatten_layer_call_and_return_conditional_losses_876623?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_flatten_1_layer_call_fn_876639?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_flatten_1_layer_call_and_return_conditional_losses_876634?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_layer2_layer_call_fn_876659?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_layer2_layer_call_and_return_conditional_losses_876652?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_multiply_layer_call_fn_876671?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_multiply_layer_call_and_return_conditional_losses_876665?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_layer3_layer_call_fn_876691?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_layer3_layer_call_and_return_conditional_losses_876684?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_concatenate_1_layer_call_fn_876704?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_concatenate_1_layer_call_and_return_conditional_losses_876698?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_prediction_layer_call_fn_876722?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_prediction_layer_call_and_return_conditional_losses_876715?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_loss_fn_0_876727?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_1_876732?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_2_876737?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_3_876742?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_4_876747?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_5_876752?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_6_876757?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
@B>
$__inference_signature_wrapper_876312
item_input
user_input
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 ?
!__inference__wrapped_model_875833?893.FGPQZ[^?[
T?Q
O?L
$?!

user_input?????????
$?!

item_input?????????
? "7?4
2

prediction$?!

prediction??????????
I__inference_concatenate_1_layer_call_and_return_conditional_losses_876698?Z?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "%?"
?
0?????????
? ?
.__inference_concatenate_1_layer_call_fn_876704vZ?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "???????????
G__inference_concatenate_layer_call_and_return_conditional_losses_876557?Z?W
P?M
K?H
"?
inputs/0????????? 
"?
inputs/1????????? 
? "%?"
?
0?????????@
? ?
,__inference_concatenate_layer_call_fn_876563vZ?W
P?M
K?H
"?
inputs/0????????? 
"?
inputs/1????????? 
? "??????????@?
E__inference_flatten_1_layer_call_and_return_conditional_losses_876634\3?0
)?&
$?!
inputs?????????
? "%?"
?
0?????????
? }
*__inference_flatten_1_layer_call_fn_876639O3?0
)?&
$?!
inputs?????????
? "???????????
E__inference_flatten_2_layer_call_and_return_conditional_losses_876534\3?0
)?&
$?!
inputs????????? 
? "%?"
?
0????????? 
? }
*__inference_flatten_2_layer_call_fn_876539O3?0
)?&
$?!
inputs????????? 
? "?????????? ?
E__inference_flatten_3_layer_call_and_return_conditional_losses_876545\3?0
)?&
$?!
inputs????????? 
? "%?"
?
0????????? 
? }
*__inference_flatten_3_layer_call_fn_876550O3?0
)?&
$?!
inputs????????? 
? "?????????? ?
C__inference_flatten_layer_call_and_return_conditional_losses_876623\3?0
)?&
$?!
inputs?????????
? "%?"
?
0?????????
? {
(__inference_flatten_layer_call_fn_876628O3?0
)?&
$?!
inputs?????????
? "???????????
B__inference_layer1_layer_call_and_return_conditional_losses_876610\89/?,
%?"
 ?
inputs?????????@
? "%?"
?
0????????? 
? z
'__inference_layer1_layer_call_fn_876617O89/?,
%?"
 ?
inputs?????????@
? "?????????? ?
B__inference_layer2_layer_call_and_return_conditional_losses_876652\FG/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? z
'__inference_layer2_layer_call_fn_876659OFG/?,
%?"
 ?
inputs????????? 
? "???????????
B__inference_layer3_layer_call_and_return_conditional_losses_876684\PQ/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? z
'__inference_layer3_layer_call_fn_876691OPQ/?,
%?"
 ?
inputs?????????
? "??????????8
__inference_loss_fn_0_876727?

? 
? "? 8
__inference_loss_fn_1_876732?

? 
? "? 8
__inference_loss_fn_2_876737?

? 
? "? 8
__inference_loss_fn_3_876742?

? 
? "? 8
__inference_loss_fn_4_876747?

? 
? "? 8
__inference_loss_fn_5_876752?

? 
? "? 8
__inference_loss_fn_6_876757?

? 
? "? ?
M__inference_mf_embedding_item_layer_call_and_return_conditional_losses_876591_3/?,
%?"
 ?
inputs?????????
? ")?&
?
0?????????
? ?
2__inference_mf_embedding_item_layer_call_fn_876597R3/?,
%?"
 ?
inputs?????????
? "???????????
M__inference_mf_embedding_user_layer_call_and_return_conditional_losses_876574_./?,
%?"
 ?
inputs?????????
? ")?&
?
0?????????
? ?
2__inference_mf_embedding_user_layer_call_fn_876580R./?,
%?"
 ?
inputs?????????
? "???????????
N__inference_mlp_embedding_item_layer_call_and_return_conditional_losses_876522_/?,
%?"
 ?
inputs?????????
? ")?&
?
0????????? 
? ?
3__inference_mlp_embedding_item_layer_call_fn_876528R/?,
%?"
 ?
inputs?????????
? "?????????? ?
N__inference_mlp_embedding_user_layer_call_and_return_conditional_losses_876505_/?,
%?"
 ?
inputs?????????
? ")?&
?
0????????? 
? ?
3__inference_mlp_embedding_user_layer_call_fn_876511R/?,
%?"
 ?
inputs?????????
? "?????????? ?
A__inference_model_layer_call_and_return_conditional_losses_876124?893.FGPQZ[f?c
\?Y
O?L
$?!

user_input?????????
$?!

item_input?????????
p

 
? "%?"
?
0?????????
? ?
A__inference_model_layer_call_and_return_conditional_losses_876163?893.FGPQZ[f?c
\?Y
O?L
$?!

user_input?????????
$?!

item_input?????????
p 

 
? "%?"
?
0?????????
? ?
A__inference_model_layer_call_and_return_conditional_losses_876385?893.FGPQZ[b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p

 
? "%?"
?
0?????????
? ?
A__inference_model_layer_call_and_return_conditional_losses_876458?893.FGPQZ[b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p 

 
? "%?"
?
0?????????
? ?
&__inference_model_layer_call_fn_876221?893.FGPQZ[f?c
\?Y
O?L
$?!

user_input?????????
$?!

item_input?????????
p

 
? "???????????
&__inference_model_layer_call_fn_876278?893.FGPQZ[f?c
\?Y
O?L
$?!

user_input?????????
$?!

item_input?????????
p 

 
? "???????????
&__inference_model_layer_call_fn_876476?893.FGPQZ[b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p

 
? "???????????
&__inference_model_layer_call_fn_876494?893.FGPQZ[b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p 

 
? "???????????
D__inference_multiply_layer_call_and_return_conditional_losses_876665?Z?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "%?"
?
0?????????
? ?
)__inference_multiply_layer_call_fn_876671vZ?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "???????????
F__inference_prediction_layer_call_and_return_conditional_losses_876715\Z[/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ~
+__inference_prediction_layer_call_fn_876722OZ[/?,
%?"
 ?
inputs?????????
? "???????????
$__inference_signature_wrapper_876312?893.FGPQZ[u?r
? 
k?h
2

item_input$?!

item_input?????????
2

user_input$?!

user_input?????????"7?4
2

prediction$?!

prediction?????????