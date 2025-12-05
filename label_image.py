"""
Module: Prediction Engine / Feature Extraction
Report Section: 5.1 Proposed Scheme & 6.2 InceptionV3 Analysis

Logic:
1. Loads the Retrained Graph (Protocol Buffer .pb)
2. Preprocesses image (Resize 299x299, Normalize Mean=128, Std=128)
3. Returns Classification Probability
"""

import os, sys
# --- FIX FOR TENSORFLOW 2.x ---
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# ------------------------------

# Suppress TensorFlow logs for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def classify_image(image_path):
    # --- Step 1: Load Pre-trained Graph ---
    model_file = "retrained_graph.pb"
    label_file = "retrained_labels.txt"
    
    # Configuration as per InceptionV3 requirements
    input_height = 299  # Report Section 5.1
    input_width = 299   # Report Section 5.1
    input_mean = 128    # Report Section 5.1
    input_std = 128     # Report Section 5.1
    
    input_layer = "Mul"
    output_layer = "final_result"

    # Load the graph
    graph = load_graph(model_file)
    
    # --- Step 2: Preprocessing (Resize & Normalize) ---
    t = read_tensor_from_image_file(image_path,
                                    input_height=input_height,
                                    input_width=input_width,
                                    input_mean=input_mean,
                                    input_std=input_std)

    # --- Step 3: Session Run & Softmax Probabilities ---
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    with tf.Session(graph=graph) as sess:
        results = sess.run(output_operation.outputs[0],
                           {input_operation.outputs[0]: t})
    
    results = results.flatten()
    labels = load_labels(label_file)

    # --- Step 4: Identify Class with Maximum Probability ---
    # We return the top result to the Flask App
    top_k = results.argsort()[-5:][::-1]
    
    # Formatting result for the UI
    top_prediction = labels[top_k[0]]
    confidence = results[top_k[0]]
    
    return f"Prediction: {top_prediction} (Confidence: {confidence:.2f})"

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()
    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)
    return graph

def read_tensor_from_image_file(file_name, input_height=299, input_width=299, input_mean=0, input_std=255):
    """
    Preprocessing function to match InceptionV3 Input Layer.
    Resizes image to 299x299 and normalizes pixel values.
    """
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(file_reader, channels=3, name='png_reader')
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(tf.image.decode_gif(file_reader, name='gif_reader'))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
    else:
        image_reader = tf.image.decode_jpeg(file_reader, channels=3, name='jpeg_reader')
        
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)
    return result

def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label