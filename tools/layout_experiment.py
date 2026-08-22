import torch 

New_Tensor_Array = [[1, 2, 3, 4], 
                    [5, 6, 7, 8], 
                    [9, 10, 11, 12]]

#Creating a new tensor and printing it's shape, Stride, if it's contiguous and Storage Adress
New_Tensor = torch.tensor(New_Tensor_Array)
print("Main Tensor: \n", New_Tensor)
print("Shape: ", tuple(New_Tensor.shape))
print("Stride: ", New_Tensor.stride())
if not New_Tensor.is_contiguous():
    New_Contiguous_Tensor = New_Tensor.contiguous()
    print("The tensor was converted to contiguous.")
else:
    print("Is Contiguous: ", New_Tensor.is_contiguous())
print("Storage Adress ", New_Tensor.untyped_storage().data_ptr())

# Creating and printing the transpose of the previous tensor
Transpose_Tensor = torch.t(New_Tensor)
print("\nTranspose Tensor: \n", Transpose_Tensor)
print("Shape: ", tuple(Transpose_Tensor.shape))
print("Stride: ", Transpose_Tensor.stride())
if not Transpose_Tensor.is_contiguous():
    Transpose_Contiguous_Tensor = Transpose_Tensor.contiguous()
    print("The was not contiguous tensor but it was converted to contiguous.")
else:
    print("Is Contiguous: ", Transpose_Tensor.is_contiguous())
print("Storage Adress ", Transpose_Tensor.untyped_storage().data_ptr())




same_adress = False
if (Transpose_Tensor.untyped_storage().data_ptr() == New_Tensor.untyped_storage().data_ptr()):
    same_adress = True
    print("Does Transpose_Tensor and New_Tensor: ", same_adress)
else:
    print("Does Transpose_Tensor and New_Tensor: ", same_adress)
    

main_contiguous = New_Tensor.contiguous()
print(New_Tensor.untyped_storage().data_ptr() == main_contiguous.untyped_storage().data_ptr())
