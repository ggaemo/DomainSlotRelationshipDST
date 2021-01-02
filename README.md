# DomainSlotRelationshipDST
Domain-slot relationship modeling using Transformers for dialogue state tracking


Script running example
python 

```
-m torch.distributed.launch 
--nproc_per_node=4 
src/main_dst.py 
--model-type albert-base-v2
--data-option split 
--batch-size 2 
--max-length 512 
--gradient-accumulation-steps 2 
--option thesis_report 
--num-train-epochs 200 
--warmup-steps 1000 
--ds-type split 
--learning-rate 1e-5 
--fp-16
```
