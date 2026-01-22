from demo.demo import app, app_css, app_theme

if __name__ == "__main__":
    # Launch the Gradio interface
    app.launch(debug=True, share=True, css=app_css, theme=app_theme)
