package main

import (
	"fmt"
	"image/color"
	"image/png"
	"os"

	"github.com/afocus/captcha"
)

func main() {
	// 创建一个验证码对象
	cap := captcha.New()

	// 设置验证码的字体
	if err := cap.SetFont("MSYH.TTC"); err != nil {
		panic(err.Error())
	}

	// 设置验证码的大小
	cap.SetSize(128, 64)

	// 设置验证码的干扰强度
	cap.SetDisturbance(captcha.MEDIUM)

	// 设置验证码的前景色
	cap.SetFrontColor(color.RGBA{255, 255, 255, 255})

	// 设置验证码的背景色
	cap.SetBkgColor(
		color.RGBA{255, 0, 0, 255},     // 红色
		color.RGBA{0, 255, 0, 255},     // 绿色
		color.RGBA{0, 0, 255, 255},     // 蓝色
		color.RGBA{255, 255, 0, 255},   // 黄色
		color.RGBA{255, 0, 255, 255},   // 紫色
		color.RGBA{0, 255, 255, 255},   // 青色
		color.RGBA{255, 165, 0, 255},   // 橙色
		color.RGBA{128, 128, 128, 255}, // 灰色
	)

	dirName := "captcha"

	// 检查目录是否存在
	if _, err := os.Stat(dirName); os.IsNotExist(err) {
		os.RemoveAll(dirName)
	}

	errDir := os.MkdirAll(dirName, 0755)
	if errDir != nil {
		panic(fmt.Sprintf("error creating directory: %v", errDir))
	}

	// 生成1000张验证码图片
	for i := 0; i < 5000; i++ {
		img, str := cap.Create(4, captcha.ALL)
		fileName := fmt.Sprintf("./"+dirName+"/%d_%s.png", i, str)
		file, err := os.Create(fileName)
		if err != nil {
			panic(err.Error())
		}

		png.Encode(file, img)
		file.Close()
	}
}
