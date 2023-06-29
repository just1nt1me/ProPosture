from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle

#Full rep counter
def get_full_reps(elbow_angles, top_full_rep_counter, bottom_full_rep_counter, full_rep_stage):
    left_elbow_angle, right_elbow_angle = elbow_angles
    average_elbow_angle=(left_elbow_angle+right_elbow_angle)/2
    if average_elbow_angle > 175 and full_rep_stage =='down':
        top_full_rep_counter +=1
    if average_elbow_angle > 175:
        full_rep_stage = "up"
    if average_elbow_angle < 60 and full_rep_stage == 'up':
        bottom_full_rep_counter +=1
    if average_elbow_angle < 60:
        full_rep_stage="down"
    return top_full_rep_counter, bottom_full_rep_counter, full_rep_stage

def get_pdf(rep_counter, top_rep_performance, bottom_rep_performance):

#GENERATING PERFORMANCE REVIEW PDF
# Sample metrics
#hands_position_score = 85.2

    # Create a PDF document
    pdf = SimpleDocTemplate("performance_review.pdf", pagesize=landscape(letter))

    # Define table data
    data = [
        ['Metrics', 'Score'],
        ['Repetitions', rep_counter],
        ['Proportion of pushups perfect at the top', '{}%'.format(top_rep_performance)],
        ['Proportion of pushups perfect at the bottom', '{}%'.format(bottom_rep_performance)],
        #['Hands Position', '{}%'.format(hands_position_score)],
    ]

    # Define table style
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.gray),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ])

    # Conditionally apply style to "perfect execution" value cell
    if top_rep_performance > 85.0:
        table_style.add('BACKGROUND', (1, 2), (1, 2), colors.green)

    if bottom_rep_performance > 85.0:
        table_style.add('BACKGROUND', (1, -1), (1, -1), colors.green)

    # Create the table and apply style
    table = Table(data,colWidths=[400,150],rowHeights=[40,25,25,25],hAlign='LEFT')
    table.setStyle(table_style)

    # Set table properties
    # table._argW[1] = 250  # Adjust the width of the table
    # table.spaceBefore = 20  # Add space before the table

    # Build the table and add it to the PDF document
    elements = [table]
    pdf.build(elements)

    return pdf
    # # Open the generated PDF with the default PDF viewer
    # subprocess.Popen(["performance_review.pdf"])
