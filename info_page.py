import streamlit as st

def generate_css(primary_color, secondary_background_color):
    css = f"""
    <style>
        body {{
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 0;
            background-color: #ffffff; /* Set background color to white */
        }}
        .container {{
            display: flex;
            flex-direction: column; /* Change flex-direction to column */
            align-items: center; /* Align items to center */
            height: 100vh;
            justify-content: center; /* Vertically center content */
        }}
        .title {{
            font-size: 2rem;
            color: {primary_color}; /* Use primary color for title */
            margin-bottom: 10px; /* Reduce margin bottom for title */
        }}
        .info-content {{
            width: 80%;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            background-color: {secondary_background_color}; /* Use secondary background color */
            margin-bottom: 20px; /* Add margin to separate sections */
        }}
    </style>
    """
    return css

def show_info_page(primary_color, secondary_background_color):
    css = generate_css(primary_color, secondary_background_color)
    st.markdown(css, unsafe_allow_html=True)

    st.title('Info Page')

    st.markdown('<div class="info-content">', unsafe_allow_html=True)
    st.header('About Polyps and Machine Learning')
    st.write("""
    Polyps are a small cluster of cells that grow on the lining of the colon. One type of polyps, known as pseudopolyps are a type of polyp that result from chronic inflammation in the colon. These polyps are found in patients with an inflammatory bowel disease (IBD), such as Ulcerative Colitis and Crohn's Disease. Pseudopolyps are harder to detect in comparison to actual polyps, as they tend to be flatter and more discrete. Machine learning (ML) is a subset of AI, which utilizes algorithms, statistical models, and “training data,” to make predictions, without being explicitly programmed. ML has applications in healthcare, as it can be used for quick and accurate diagnoses. This website uses a ML model with an avergae accuracy of 92% that can detect polyps in the colon. 
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="info-content">', unsafe_allow_html=True)
    st.header('About Inflammatory Bowel Disease')
    st.write("""
    Inflammatory bowel disease (IBD) is a group of chronic inflammatory conditions of the gastrointestinal tract, primarily ulcerative colitis and Crohn's disease. These conditions are caused by inflammation of the digestive tract lining, leading to symptoms such as abdominal pain, diarrhea, rectal bleeding, fatigue, and weight loss. While the exact cause of IBD is unknown, it is believed to involve a combination of genetic, environmental, and immune factors. Treatment typically involves medications to reduce inflammation and manage symptoms, as well as lifestyle changes and, in some cases, surgery. Psuedopolyps are linked with severe IBD and, as they can be hard to detect, they are commonly not identified during colonoscopies. Therefore, being able to identify these psuedopolyps earlier will allow for early treatment and reduce symptoms in these patients. 
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="info-content">', unsafe_allow_html=True)
    st.header('About Diya Ramakrishnan')

    st.subheader('Background')
    st.write("""
    Hello! My name is Diya Ramakrishnan, and I am a freshman at Saginaw Arts and Science Academy. My favorite subjects include science, mathematics, engineering, and coding.
    """)
    
    st.subheader('Inspiration')
    st.write("""
    My passion for coding and technology stems from a desire to make a positive impact in the world. I am particularly motivated by the challenges faced by individuals suffering from inflammatory bowel disease (IBD), as there are many people that I have seen struggle from IBD. By creating a ML model, I hope that this model can allow for early treatment and prevent the severity of IBD from increasing.
    """)
    
    st.subheader('Mission')
    st.write("""
    Through this website, I aim to utilize machine learning and technology to assist healthcare professionals in the early detection of polyps during colonoscopies. This way, doctors can improve patient outcomes and contribute to the early diagnosis and treatment of inflammatory bowl disease and other colorectal conditions.
    """)
    
    st.subheader('Contact')
    st.write("""
    I'm excited to continue my journey in coding and contribute to advancements in healthcare technology. If you have any questions, suggestions, or just want to say hello, feel free to reach out to me at diyaramakrishnan009@gmail.com.
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)
