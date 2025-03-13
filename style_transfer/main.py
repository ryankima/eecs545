from ST import style_transfer
if __name__ == "__main__":
    # Define paths
    content_path = "unnamed-2.jpg"
    style_path = "test-content/basal_cell_carcinoma.jpg"
    output_dir = "new_test"

    # Call the function
    style_transfer(content_path, style_path, output_dir)