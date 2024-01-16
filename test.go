package main

import (
	"fmt"
)

// Tensor es una estructura simple para almacenar datos de tipo int
type Tensor struct {
	shape []int
	data  interface{}
}

// NewTensor crea un nuevo tensor con la forma y los datos proporcionados
func NewTensor(shape []int, data interface{}) *Tensor {
	if len(shape) == 0 {
		panic("La forma del tensor no puede estar vacía")
	}

	return &Tensor{shape, data}
}

// Reshape modifica la forma del tensor
func (t *Tensor) Reshape(newShape []int) *Tensor {
	newSize := 1
	for _, dim := range newShape {
		newSize *= dim
	}

	if newSize != len(t.Flatten()) {
		panic("El nuevo tamaño después de la reformulación no coincide con el tamaño original")
	}

	return &Tensor{newShape, t.data}
}

// HadamardProduct realiza el producto de Hadamard entre dos tensores
func (t *Tensor) HadamardProduct(other *Tensor) *Tensor {
	if !shapeEqual(t.shape, other.shape) {
		panic("Los tensores deben tener la misma forma para realizar el producto de Hadamard")
	}

	resultData := make([]int, len(t.Flatten()))
	for i := range t.Flatten() {
		resultData[i] = t.Flatten()[i] * other.Flatten()[i]
	}

	return &Tensor{t.shape, resultData}
}

// IndexSelect selecciona datos de una dimensión específica según los índices proporcionados
func (t *Tensor) IndexSelect(dim int, indices []int) *Tensor {
	switch data := t.data.(type) {
	case []int:
		return &Tensor{t.shape, indexSelect1D(data, dim, indices)}
	case [][]int:
		return &Tensor{t.shape, indexSelect2D(data, dim, indices)}
	default:
		panic("Tipo de datos no compatible para IndexSelect")
	}
}

// Flatten devuelve los datos del tensor aplanados en una sola dimensión
func (t *Tensor) Flatten() []int {
	switch data := t.data.(type) {
	case []int:
		return data
	case [][]int:
		flattened := make([]int, 0, len(data)*len(data[0]))
		for _, row := range data {
			flattened = append(flattened, row...)
		}
		return flattened
	default:
		panic("Tipo de datos no compatible para Flatten")
	}
}

// shapeEqual verifica si las formas de dos tensores son iguales
func shapeEqual(shape1, shape2 []int) bool {
	if len(shape1) != len(shape2) {
		return false
	}

	for i := range shape1 {
		if shape1[i] != shape2[i] {
			return false
		}
	}

	return true
}

// indexSelect1D selecciona datos de un tensor unidimensional según los índices proporcionados
func indexSelect1D(tensor []int, dim int, indices []int) []int {
	result := make([]int, len(indices))
	for i, index := range indices {
		result[i] = tensor[index]
	}
	return result
}

// indexSelect2D selecciona datos de un tensor bidimensional según los índices proporcionados
func indexSelect2D(tensor [][]int, dim int, indices []int) [][]int {
	result := make([][]int, len(indices))
	for i, index := range indices {
		result[i] = make([]int, len(tensor[index]))
		copy(result[i], tensor[index])
	}
	return result
}

func main() {
	// Ejemplo de uso
	tensorData := [][]int{{1, 2}, {3, 4}}
	tensorShape := []int{2, 2}
	tensor := NewTensor(tensorShape, tensorData)

	// Operación Reshape
	newShape := []int{4}
	reshapedTensor := tensor.Reshape(newShape)
	fmt.Println("Reshape:", reshapedTensor)

	// Operación Hadamard Product
	otherData := [][]int{{2, 2}, {2, 2}}
	otherTensor := NewTensor(tensorShape, otherData)
	hadamardProduct := tensor.HadamardProduct(otherTensor)
	fmt.Println("Hadamard Product:", hadamardProduct)

	// Operación Index Select
	indices := []int{0, 0, 1, 1}
	selectedTensor := tensor.IndexSelect(0, indices)
	fmt.Println("Index Select:", selectedTensor)
}
