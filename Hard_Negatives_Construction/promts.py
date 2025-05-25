PROMPT_CAPTION_GENERATION = """Generate 10 slightly modified negative caption examples for the given geometric image caption and code. These negative examples will enhance the model's image-text alignment ability through contrastive learning. Ensure modifications are meaningful yet still consistent with the geometric composition.

Modification areas include:
         1. Specific geometric concepts, e.g., square ABCD → rectangle ABCD; circle o1 diameter → oval o1 diameter; equilateral triangle → isosceles triangle.
         2. Letter indicator modifications. Change at least two letters, avoiding same start-end sequences (ABCD → CDAB is not allowed). Example: square ABCD → square CBDA is valid; AC → CA is invalid (only two letters).
         3. Numerical values, e.g., square ABCD with side length 8.0 → side length 6.0.
         4. Similar verb substitutions (triangle similar → triangle congruent), but avoid major changes like parallel → perpendicular.
         5. Note: Changing notation markers is incorrect, and edge AB → segment BA is invalid (edges and segments are equivalent).

Requirements:
         1. Don't introduce letters not present in the original caption (e.g., trapezoid GHIJ → trapezoid XYZO is invalid).
         2. Distribute modifications evenly throughout the caption, not just in one section.

Original caption:
```
{}
```
Geometric code:
```
{}
```
Output format:
```
         1. caption1 \n 
         2. caption2 \n  
         3. caption3 \n 
         4. caption4 \n 
         5. caption5 \n 
         6. caption6 \n 
         7. caption7 \n 
         8. caption8 \n 
         9. caption9 \n 
         10. caption10 \n 
```
"""

NEGATIVE_CAPTION_CODE_GENERATION = """Modify the provided geometric image code according to the negative caption description:

1. Ensure the output image matches the negative caption description and save it as "question.png"
2. Maintain the format of the original geometric code
3. Keep the modified geometric figure within the frame boundaries
4. Don't include plt.show() at the end of the code; instead save the file as "question.png"

Negative caption description:
```
{}
```
Geometric code:
```
{}
```"""