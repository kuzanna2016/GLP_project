# Annotation guidelines:  
Annotate clusters of noun phrases referring to the same object/group of objects. 
There should be at least two co-referenced nps in the cluster 
(so no solo nps or solo pronouns, there should be at least two spans in the cluster). 
The indexes are the indexes of the start and end of the span, 
if there is only one token in the span the start and end are the same.

Some tricky parts:

## 1. Plural:  
- Sometimes the pronoun can refer to a group that has not been named. 
For now I don't know what to do with it, I don't consider it a co-reference. 
I decided that nps in the questions like _other things_ don't count as references, if you don't agree let's discuss. 
Example from 0,3:	
  - 81.are 82.there __83.other 84.things__ 85.in 86.the 87.picture 88.that 89.share 90.the 91.same 92.size 93.with 94.the 95.aforementioned 96.small 97.thing 98.?
  - 100.yes
  - 102.any 103.cylinders 104.among __105.them__ 106.?
- Other times, in the caption, the group is named and then it is easy to co-reference it.
Example from 3,2:
  - 0.there 1.are __2.3 3.cylinders__ 4.in 5.the 6.picture 7..
  - 9.any 10.matt 11.##e 12.objects 13.among __14.them__ 15.?
- If there is a counting question and a reference to the group it counted, use the answer as a reference.
Example from 100,0:	
  - 73.how 74.many 75.other 76.objects 77.in 78.the 79.picture 80.have 81.the 82.same 83.shape 84.as 85.the 86.previous 87.green 88.object 89.?
  - __91.2__
  - 93.are 94.there 95.any 96.metallic 97.objects 98.among __99.them__ 100.?
  - 102.yes
## 2. Counting
When an answer to a counting question is 1, it can be used as a reference in the next question.
I decided again to use the answer as the reference, not the group in question (_yellow objects_ in the example). 
Example from 104,3:	
  - Q1: 14.what 15.is 16.the 17.number 18.of __19.yellow 20.objects__ 21.in 22.the 23.image 24., 25.if 26.present 27.?
  - A: __29.1__
  - Q2: 31.what 32.about __33.its__ 34.shape 35.?
  - A: 37.cube

## 3. Any
Kind of related to counting, existence questions can also lead to a reference, but in this case,
I decided to use the group in question as a reference.
Example from 100,0:	
-   73.how 74.many 75.other 76.objects 77.in 78.the 79.picture 80.have 81.the 82.same 83.shape 84.as 85.the 86.previous 87.green 88.object 89.?
- 	_91.2_
- 	93.are 94.there __95.any 96.metallic 97.objects__ 98.among _99.them_ 100.?
- 	102.yes
- 	...
- 	124.what 125.size 126.is __127.that 128.metal 129.object__ 130.?

## 4. Resolving pronoun with an existing group requires visual reference.
Sometimes it is not obvious just from the text that the pronoun refers to the already existing group, 
so use the objects description on top, or the picture to guide you.
Examples:
- From 3,0: 74, 80, 83 - look like a new cluster, but are actually the named metallic sphere
![](/home/kuzya/Desktop/uni/GLP/visual_reference30.png)
- From 0,3 - 6 round
- From 3,3 - 9 round


3.exactly 4.one 5.ball

C: __0.no 1.other 2.cylinder__ 3.except 4.for __5.one__ 6..
__0.no 1.other 2.cylinder__ 3.except 4.for 5.exactly __6.one__ 7..
C: __0.no 1.other __2.cy 3.##an 4.object__ 5.except 6.for 7.exactly __8.one__ 9..


Vispro and Visdial notes: 
1. in vispro no cluster of length 1
2. how many questions don't go into cluster
3. answer to a count question can go into cluster (visdial 173)
	- Q: 25.how 26.many 27.people 28.are 29.there
	- A: 31.there 32.are __33.4__
4. 'any' attaches to an np
	can you see any writing on their shirts - 'any writing'
	are there any planes - 'any planes'
