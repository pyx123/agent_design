#!/bin/bash
# 推送代码到GitHub的脚本

# 请替换以下变量为您的实际信息
GITHUB_USERNAME="yourusername"
REPO_NAME="devops-agent"

echo "准备推送代码到 GitHub..."
echo "请确保您已经在 GitHub 上创建了仓库: https://github.com/${GITHUB_USERNAME}/${REPO_NAME}"
echo ""
echo "如果您还没有创建仓库，请先访问 GitHub 创建一个新仓库。"
echo "创建时不要初始化 README、.gitignore 或 LICENSE。"
echo ""
read -p "按 Enter 继续，或按 Ctrl+C 取消..."

# 添加远程仓库
echo "添加远程仓库..."
git remote add origin https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git

# 推送代码
echo "推送代码到 GitHub..."
git push -u origin main

echo ""
echo "代码推送完成！"
echo "您可以访问 https://github.com/${GITHUB_USERNAME}/${REPO_NAME} 查看您的仓库。"