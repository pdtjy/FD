# FastDeploy 文档版本管理指南

本文档介绍如何使用 Mike 工具管理 FastDeploy 文档的多个版本。

## Mike 简介

Mike 是 MkDocs 的版本管理工具，可以：
- 为每个版本生成独立的静态文档
- 提供版本选择器界面
- 设置版本别名（如 latest, stable）
- 保持历史版本的完整性

## 安装依赖

```bash
pip install -r requirements.txt
```

## 基本使用命令

### 1. 首次部署文档版本

```bash
# 部署第一个版本（例如 v1.0.0）
mike deploy v1.0.0

# 部署并设置为最新版本
mike deploy --update-aliases v1.0.0 latest
```

### 2. 添加新版本

```bash
# 添加新版本 v2.0.0
mike deploy v2.0.0

# 更新 latest 别名指向最新版本
mike deploy --update-aliases v2.0.0 latest

# 也可以设置 stable 别名指向稳定版本
mike deploy --update-aliases v2.0.0 stable
```

### 3. 设置默认版本

```bash
# 设置默认重定向版本（用户访问根目录时的默认版本）
mike set-default latest
```

### 4. 查看现有版本

```bash
# 列出所有已部署的版本
mike list
```

### 5. 删除旧版本

```bash
# 删除特定版本
mike delete v1.0.0

# 删除版本及其别名
mike delete v1.0.0 --all
```

## 推荐的版本管理策略

### 语义化版本控制

建议使用语义化版本号：
- `v1.0.0` - 主要版本
- `v1.1.0` - 次要版本
- `v1.1.1` - 补丁版本

### 别名策略

- `latest` - 始终指向最新发布的版本
- `stable` - 指向稳定的生产版本
- `dev` - 开发版本（可选）

### 完整部署示例

```bash
# 1. 部署第一个稳定版本
mike deploy v1.0.0
mike set-default v1.0.0
mike deploy --update-aliases v1.0.0 stable

# 2. 发布次要更新
mike deploy v1.1.0
mike deploy --update-aliases v1.1.0 latest

# 3. 发布主要版本更新
mike deploy v2.0.0
mike set-default v2.0.0
mike deploy --update-aliases v2.0.0 stable latest
```

## 与 CI/CD 集成

可以在 GitHub Actions 等 CI 系统中自动部署：

```yaml
# .github/workflows/docs.yml
name: Deploy Documentation
on:
  release:
    types: [published]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
          
      - name: Install dependencies
        run: pip install -r requirements.txt
        
      - name: Deploy with Mike
        run: |
          mike deploy ${{ github.ref_name }} latest
          mike set-default latest
```

## 注意事项

1. **版本命名**: 避免使用空格和特殊字符
2. **别名管理**: 合理使用别名，避免混淆
3. **备份**: 定期备份文档源文件
4. **测试**: 在部署前测试各个版本的兼容性

## 故障排除

### 常见问题

1. **版本不显示**: 检查 mkdocs.yml 配置是否正确
2. **构建失败**: 确认所有依赖已安装
3. **路径错误**: 检查文档路径和链接

### 获取帮助

- Mike 官方文档: https://github.com/jimporter/mike
- MkDocs 文档: https://www.mkdocs.org/
- FastDeploy 文档: 参考本项目其他文档

---

通过合理的版本管理，可以确保用户能够访问适合其 FastDeploy 版本的文档，同时保持历史文档的可访问性。