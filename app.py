import os
import tempfile
import gradio as gr
# pyrefly: ignore [missing-import]
from transformers import pipeline

# ====================================================
# Load Model with Lazy Loading & Mock Fallback
# ====================================================

generator_pipeline = None

def get_generator():
    global generator_pipeline
    if generator_pipeline is None:
        try:
            print("Loading text generation model 'distilgpt2'...")
            generator_pipeline = pipeline(
                "text-generation",
                model="distilgpt2"
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Mock fallback in case torch/transformers has environment issues
            class MockGenerator:
                def __call__(self, prompt, **kwargs):
                    return [{"generated_text": prompt + "\n[Mock Story Content: Could not load the local model. Please verify torch/transformers are installed properly.]"}]
            generator_pipeline = MockGenerator()
    return generator_pipeline

# ====================================================
# Story Generator Function
# ====================================================

def generate_story(topic, genre, length, character_name, character_role, setting, history_state):
    # Formulate a structured prompt to guide the model
    prompt_parts = []
    
    # Starting setup
    prompt_parts.append(f"A {genre} story about: {topic}.")
    
    # Character info
    if character_name or character_role:
        char_desc = "The main character is "
        if character_name:
            char_desc += character_name
            if character_role:
                char_desc += f", a {character_role}"
        else:
            char_desc += f"a {character_role}"
        prompt_parts.append(char_desc + ".")
        
    # Setting info
    if setting:
        prompt_parts.append(f"The story is set in {setting}.")
        
    # Introduce the narrative transition
    prompt_parts.append("Once upon a time,")
    
    prompt = " ".join(prompt_parts)

    # Determine token length
    if length == "Short":
        tokens = 120
    elif length == "Medium":
        tokens = 220
    else:
        tokens = 350

    try:
        generator = get_generator()
        result = generator(
            prompt,
            max_new_tokens=tokens,
            do_sample=True,
            temperature=0.85,
            top_k=40,
            top_p=0.90,
            clean_up_tokenization_spaces=True
        )
        generated_story = result[0]["generated_text"]
    except Exception as e:
        generated_story = f"An error occurred during generation: {str(e)}"

    # Save to history list
    new_story_info = {
        "topic": topic,
        "genre": genre,
        "length": length,
        "story": generated_story
    }
    history_state.append(new_story_info)

    # Update drop-downs or other UI components in history tab
    history_choices = [f"#{i+1}: [{item['genre']}] {item['topic'][:30]}..." for i, item in enumerate(history_state)]
    
    return generated_story, history_state, gr.Dropdown(choices=history_choices, value=history_choices[-1])

# ====================================================
# Helper Functions for UI
# ====================================================

def download_story(story_text):
    if not story_text or story_text.strip() == "":
        return None
    
    # Create temp file
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, "generated_story.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(story_text)
    
    return file_path

def load_selected_history(selected_index_str, history):
    if not selected_index_str or not history:
        return ""
    try:
        # Extract index from format: "#1: [Genre] Topic..."
        idx = int(selected_index_str.split(":")[0].replace("#", "")) - 1
        return history[idx]["story"]
    except Exception:
        return ""

# ====================================================
# UI Custom Styling & Layout
# ====================================================

theme = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="indigo",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Outfit"), "ui-sans-serif", "system-ui", "sans-serif"],
)

custom_css = """
.title-container {
    text-align: center;
    padding: 2.5rem 1.5rem;
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
    border-radius: 16px;
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0 10px 25px rgba(99, 102, 241, 0.15);
}
.title-container h1 {
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    color: white !important;
    margin-bottom: 0.5rem;
    letter-spacing: -0.5px;
}
.title-container p {
    font-size: 1.2rem !important;
    opacity: 0.95;
    font-weight: 300;
}
.generate-btn {
    background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(139, 92, 246, 0.25) !important;
    transition: all 0.2s ease-in-out !important;
}
.generate-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(139, 92, 246, 0.4) !important;
}
.generate-btn:active {
    transform: translateY(1px) !important;
}
.accent-card {
    border: 1px solid rgba(226, 232, 240, 0.8) !important;
    background-color: #fcfdff !important;
    border-radius: 12px !important;
}
"""

examples = [
    ["A robot exploring a forgotten village", "Science Fiction", "Short", "Sparky", "mechanic drone", "Old Forest Village"],
    ["A secret passage hidden inside a magic library", "Fantasy", "Medium", "Lyra", "apprentice mage", "The Grand Archives"],
    ["A cold fog rolling into a haunted mansion", "Horror", "Short", "Arthur", "skeptical investigator", "Blackwood Manor"],
    ["Finding a mythical lost gold city in the jungle", "Adventure", "Long", "Captain Drake", "explorer", "Amazon Basin"],
    ["A detective solving a jewelry heist on a luxury train", "Mystery", "Medium", "Inspector Poirot", "sleuth", "Orient Express"]
]

# ====================================================
# App Setup
# ====================================================

with gr.Blocks(
    title="AI Story Generator"
) as demo:

    # App-level story history state
    history_state = gr.State([])

    # Header section
    gr.HTML(
        """
        <div class="title-container">
            <h1>📖 AI Story Generator</h1>
            <p>Craft compelling, imaginative tales instantly with open-source AI. No API Keys required.</p>
        </div>
        """
    )

    with gr.Tabs():
        
        # Generator Tab
        with gr.TabItem("✨ Create a Story"):
            with gr.Row():
                with gr.Column(scale=1, elem_classes="accent-card"):
                    gr.Markdown("### 🎨 Story Settings")
                    
                    topic = gr.Textbox(
                        label="Story Prompt / Topic",
                        placeholder="What should the story be about? (e.g. A dragon guarding a golden key)",
                        lines=3
                    )
                    
                    with gr.Row():
                        genre = gr.Dropdown(
                            choices=["Fantasy", "Science Fiction", "Adventure", "Mystery", "Horror", "Comedy"],
                            value="Fantasy",
                            label="Genre"
                        )
                        
                        length = gr.Radio(
                            choices=["Short", "Medium", "Long"],
                            value="Short",
                            label="Length"
                        )
                    
                    with gr.Accordion("👤 Add Character & Setting details (Optional)", open=False):
                        with gr.Row():
                            character_name = gr.Textbox(
                                label="Character Name",
                                placeholder="e.g. Eldrin"
                            )
                            character_role = gr.Textbox(
                                label="Character Role/Class",
                                placeholder="e.g. wizard, pilot, detective"
                            )
                        setting = gr.Textbox(
                            label="Setting / Location",
                            placeholder="e.g. a floating island, Victorian London"
                        )
                        
                    generate_btn = gr.Button(
                        "Generate Story ✨",
                        elem_classes="generate-btn"
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### 📝 Generated Story")
                    output = gr.Textbox(
                        label="Resulting Story",
                        lines=15,
                        interactive=False,
                        placeholder="Your story will appear here once you hit Generate..."
                    )
                    
                    with gr.Row():
                        download_btn = gr.Button("💾 Download as File")
                        download_file = gr.File(label="Downloadable File", visible=False)

            # Examples Section
            gr.Examples(
                examples=examples,
                inputs=[
                    topic,
                    genre,
                    length,
                    character_name,
                    character_role,
                    setting
                ]
            )

        # History Tab
        with gr.TabItem("📜 Story History Log"):
            gr.Markdown("### 🕒 Session History")
            history_dropdown = gr.Dropdown(
                label="Select a previously generated story",
                choices=[],
                interactive=True
            )
            history_display = gr.Textbox(
                label="Story Content",
                lines=15,
                interactive=False,
                placeholder="Select a story from the dropdown above to display it here..."
            )

        # About Tab
        with gr.TabItem("ℹ️ About the App"):
            gr.Markdown(
                """
                ### How it works
                This story generator uses Hugging Face's lightweight **distilgpt2** model (a distilled version of GPT-2) running locally inside PyTorch.
                
                - **No API Keys**: The model runs locally in your Python environment.
                - **Fast Inference**: Distilled parameters ensure quick response times even on CPUs.
                - **Privacy First**: Your prompts and stories never leave your local workspace unless you publish the share link.
                
                ### Customizing Generation
                To guide the text completion model more effectively:
                1. Provide a descriptive topic.
                2. Expand the character/setting details accordion.
                3. Choose a length that fits your goal.
                """
            )

    # Click interactions
    generate_btn.click(
        generate_story,
        inputs=[
            topic,
            genre,
            length,
            character_name,
            character_role,
            setting,
            history_state
        ],
        outputs=[
            output,
            history_state,
            history_dropdown
        ]
    )

    # Download interaction
    download_btn.click(
        download_story,
        inputs=output,
        outputs=download_file
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=download_file
    )

    # History display interaction
    history_dropdown.change(
        load_selected_history,
        inputs=[history_dropdown, history_state],
        outputs=history_display
    )

if __name__ == "__main__":
    demo.launch(theme=theme, css=custom_css,share=True)
