from pathlib import Path
import re
import chardet

class ContentExporters:
    """Exporters for content in various formats."""
    
    @staticmethod
    def clean_text(text):
        """Clean text to be compatible with Latin-1 encoding."""
        # Detect text encoding
        if isinstance(text, bytes):
            detected = chardet.detect(text)
            encoding = detected['encoding'] or 'utf-8'
            text = text.decode(encoding, errors='replace')
        
        # Common Unicode character replacements
        replacements = {
            '–': '-',  # en dash
            '—': '--',  # em dash
            '"': '"',  # left double quote
            '"': '"',  # right double quote
            ''': "'",  # left single quote
            ''': "'",  # right single quote
            '…': '...',  # ellipsis
            '•': '*',  # bullet point
            '→': '->',  # right arrow
            '←': '<-',  # left arrow
            '±': '+/-',  # plus-minus
            '×': 'x',  # multiplication
            '÷': '/',  # division
            '°': ' deg',  # degree
            'Ω': 'Ohm',  # ohm symbol
            'α': 'alpha',  # alpha
            'β': 'beta',  # beta
            'γ': 'gamma',  # gamma
            'δ': 'delta',  # delta
            'μ': 'mu',  # mu
            'π': 'pi',  # pi
            '∞': 'infinity',  # infinity
            '©': '(c)',  # copyright
            '®': '(R)',  # registered trademark
            '™': '(TM)',  # trademark
            '€': 'EUR',  # euro
            '£': 'GBP',  # pound
            '¥': 'JPY',  # yen
            '§': 'S',  # section
            '¶': 'P',  # paragraph
            '†': '+',  # dagger
            '‡': '++',  # double dagger
            '•': '*',  # bullet
            '…': '...',  # horizontal ellipsis
            '′': "'",  # prime
            '″': '"',  # double prime
            '‴': '"',  # triple prime
            '‵': '`',  # reversed prime
            '‶': '``',  # reversed double prime
            '‷': '```',  # reversed triple prime
            '‸': '^',  # caret
            '※': '*',  # reference mark
            '‼': '!!',  # double exclamation
            '⁇': '??',  # double question
            '⁈': '?!',  # question exclamation
            '⁉': '!?',  # exclamation question
            '⁎': '*',  # low asterisk
            '⁏': ';',  # reversed semicolon
            '⁐': 'P',  # close up
            '⁑': '**',  # two asterisks
            '⁒': '%',  # commercial minus
            '⁓': '~',  # swung dash
            '⁔': '_',  # undertie
            '⁕': '*',  # flower punctuation
            '⁖': '...',  # three dot punctuation
            '⁗': '....',  # four dot punctuation
            '⁘': '-',  # five dot punctuation
            '⁙': '.....',  # six dot punctuation
            '⁚': '..',  # two dot punctuation
            '⁛': '....',  # four dot mark
            '⁜': '...',  # dotted cross
            '⁝': '...',  # tricolon
            '⁞': '....',  # vertical four dots
        }
        
        # Replace known Unicode characters
        for unicode_char, ascii_char in replacements.items():
            text = text.replace(unicode_char, ascii_char)
        
        # Remove any remaining non-Latin-1 characters
        return ''.join(char for char in text if ord(char) < 256)

    @staticmethod
    def export_as_pdf(content, filename_base, output_dir):
        """Export content as PDF.
        
        Args:
            content: The formatted content to export
            filename_base: Base filename without extension
            output_dir: Directory to save the output
            
        Returns:
            Path to the generated file or None if failed
        """
        try:
            from fpdf import FPDF
            
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            
            # Split content into lines and add to PDF
            for line in content.split('\n'):
                try:
                    # Clean the line and skip empty lines and style-related content
                    cleaned_line = ContentExporters.clean_text(line)
                    if cleaned_line.strip() and not cleaned_line.strip().startswith(('style', '/*', '*/')):
                        pdf.multi_cell(0, 10, cleaned_line)
                except Exception as e:
                    print(f"Warning: Could not process line: {str(e)}")
                    continue
            
            output_path = output_dir / f"{filename_base}.pdf"
            pdf.output(str(output_path))
            return output_path
        except ImportError as e:
            print(f"Required package not installed: {str(e)}")
            print("Please install required packages with: pip install fpdf chardet")
            return None
        except Exception as e:
            print(f"Error creating PDF: {str(e)}")
            return None
    
    @staticmethod
    def export_as_html(content, filename_base, output_dir):
        """Export content as HTML.
        
        Args:
            content: The formatted content to export
            filename_base: Base filename without extension
            output_dir: Directory to save the output
            
        Returns:
            Path to the generated file
        """
        import markdown
        from datetime import datetime
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract title from content or use filename
        title = filename_base.replace("_", " ").title()
        title_match = re.search(r'^# (.+)$', content, re.MULTILINE)
        if title_match:
            title = title_match.group(1)
        
        # Process markdown with minimal extensions
        md_content = markdown.markdown(
            content,
            extensions=['markdown.extensions.tables']
        )
        
        # Current date for the document
        current_date = datetime.now().strftime("%B %d, %Y")
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
</head>
<body>
    <header>
        <h1>{title}</h1>
        <div>Generated on {current_date}</div>
    </header>
    
    <main>
        {md_content}
    </main>
    
    <footer>
        <p>Source document: {filename_base}</p>
    </footer>
</body>
</html>
"""
        
        output_path = output_dir / f"{filename_base}.html"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        return output_path 