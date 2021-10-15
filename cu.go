package main

import (
	"fmt"
	"log"
	"os"
	"strconv"
	"time"

	"gonum.org/v1/gonum/blas"
	"gorgonia.org/cu"
	cublas "gorgonia.org/cu/blas"
	"gorgonia.org/tensor"
)

var attrs = []cu.DeviceAttribute{
	cu.GpuOverlap,
	cu.GlobalMemoryBusWidth,
	cu.AsyncEngineCount,
}

func calcMemsize(dt tensor.Dtype, shape tensor.Shape) int64 {
	return int64(shape.TotalSize() * 4)
}

func main() {
	if len(os.Args) != 4 {
		fmt.Println("./cu m n k")
		os.Exit(1)
	}
	m, _ := strconv.Atoi(os.Args[1])
	n, _ := strconv.Atoi(os.Args[2])
	k, _ := strconv.Atoi(os.Args[3])

	t := time.Now()
	dev := cu.Device(0)
	printDevice(dev)
	ctx, err := dev.MakeContext(cu.SchedAuto)
	if err != nil {
		log.Fatal(err)
	}
	defer ctx.Destroy()

	fmt.Println("initialized context:", time.Since(t))
	t = time.Now()
	dt := tensor.Float32
	s0 := tensor.Shape{m, n}
	s1 := tensor.Shape{n, k}
	s2 := tensor.Shape{m, k}

	memsize0 := calcMemsize(dt, s0)
	mem0, err := cu.MemAllocManaged(memsize0, cu.AttachGlobal)
	if err != nil {
		log.Fatal(err)
	}
	mat0 := tensor.New(tensor.Of(dt), tensor.WithShape(s0...), tensor.FromMemory(uintptr(mem0), uintptr(memsize0)))
	d0 := mat0.Data().([]float32)
	for i := range d0 {
		d0[i] = float32(i + 1)
	}
	//fmt.Printf("A: \n%#v\n", mat0)

	memsize1 := calcMemsize(dt, s1)
	mem1, err := cu.MemAllocManaged(memsize1, cu.AttachGlobal)
	if err != nil {
		log.Fatal(err)
	}
	mat1 := tensor.New(tensor.Of(dt), tensor.WithShape(s1...), tensor.FromMemory(uintptr(mem1), uintptr(memsize1)))
	d1 := mat1.Data().([]float32)
	for i := range d1 {
		d1[i] = float32(i + 1)
	}
	//fmt.Printf("B: \n%#v\n", mat1)

	memsize2 := calcMemsize(dt, s2)
	mem2, err := cu.MemAllocManaged(memsize2, cu.AttachGlobal)
	if err != nil {
		log.Fatal(err)
	}
	mat2 := tensor.New(tensor.Of(dt), tensor.WithShape(s2...), tensor.FromMemory(uintptr(mem2), uintptr(memsize2)))
	d2 := mat2.Data().([]float32)
	for i := range d2 {
		d2[i] = float32(i + 1)
	}
	//fmt.Printf("C: \n%#v\n", mat2)

	fmt.Println("created matrices:", time.Since(t))
	t = time.Now()

	impl := cublas.New()

	fmt.Println("created cublas impl:", time.Since(t))
	t = time.Now()

	// m := s0[0]
	// k := s0[1]
	// n := s1[1]

	lda := mat0.Strides()[0]
	//ldb := mat1.Strides()[0]
	ldc := mat2.Strides()[0]
	alpha := float32(1.0)
	beta := float32(0.0)
	impl.Sgemm(blas.NoTrans, blas.NoTrans, k, m, n, alpha, d1, ldc, d0, lda, beta, d2, ldc)
	if err := cu.Synchronize(); err != nil {
		log.Fatal(err)
	}
	// fmt.Printf("C: \n%#v\n", mat2)
	fmt.Println("gemm:", time.Since(t))
	t = time.Now()

	cu.MemFree(mem0)
	cu.MemFree(mem1)
	cu.MemFree(mem2)
}

func printDevice(dev cu.Device) {
	devName, err := dev.Name()
	if err != nil {
		panic(err)
	}

	attrValues, err := dev.Attributes(attrs...)
	if err != nil {
		panic(err)
	}

	fmt.Println("Device:", devName)
	fmt.Println(" is gpu:", dev.IsGPU())
	fmt.Println(" overlap:", attrValues[0])
	fmt.Println(" memory bus:", attrValues[1])
	fmt.Println(" async engines:", attrValues[2])
}

// func main() {

// 	dev := cu.Device(0)
// 	ctx, err := dev.MakeContext(cu.SchedAuto)
// 	if err != nil {
// 		log.Fatal(err)
// 	}
// 	defer cu.DestroyContext(&ctx)

// 	dev, err := cu.GetDevice(1)
// 	if err != nil {
// 		panic(err)
// 	}
// 	devName, err := dev.Name()
// 	if err != nil {
// 		panic(err)
// 	}

// 	attrValues, err := dev.Attributes(attrs...)
// 	if err != nil {
// 		panic(err)
// 	}

// 	fmt.Println("Device:", devName)
// 	fmt.Println(" is gpu:", dev.IsGPU())
// 	fmt.Println(" overlap:", attrValues[0])
// 	fmt.Println(" memory bus:", attrValues[1])
// 	fmt.Println(" async engines:", attrValues[2])
// }
