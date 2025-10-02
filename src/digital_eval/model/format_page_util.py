"""format_page_util module"""
from pathlib import PurePath
from typing import List, Tuple
from xml.dom.minidom import Document, Element, parse

from digital_eval.model.digital_object_model import DigitalObjectTree
from digital_eval.model.common import DigitalObjectException, DigitalObjectLevel, \
    DigitalObjectTreeOCRFileFormat


class FormatPageUtil:
    """helper methods for parsing Page XML files"""

    @staticmethod
    def extract_data(path: str) -> DigitalObjectTree:
        document: Document = parse(path)
        doc_root: Element = document.documentElement
        ns = doc_root.namespaceURI
        pages = doc_root.getElementsByTagNameNS(ns, 'Page')
        if len(pages) != 1:
            raise DigitalObjectException(f"No unique page for {path}")
        page_one = pages[0]
        page_width = int(page_one.getAttribute('imageWidth'))
        page_height = int(page_one.getAttribute('imageHeight'))
        top_piece = DigitalObjectTree(
            page_one.getAttribute('imageFilename'),
            page_one,
            document=document,
            file_format=DigitalObjectTreeOCRFileFormat.PAGE,
            file_path=PurePath(path)
        )
        top_piece.level = DigitalObjectLevel.PAGE
        top_piece.dimensions = [[0, 0], [page_width, 0],
                                [page_width, page_height], [0, page_height]]
        regions = doc_root.getElementsByTagNameNS(ns, 'TextRegion')
        regions.extend(doc_root.getElementsByTagNameNS(ns, 'TableCell'))
        # no regions are considered to be reasonable
        # don't raise exception, it's an empty page
        if len(regions) < 1:
            return top_piece

        # inspect *all* regions
        region_pieces: List[DigitalObjectTree] = []
        for region in regions:
            this_piece: DigitalObjectTree = FormatPageUtil.__from_text_element(region, top_piece, ns)
            # go into details
            page_lines = region.getElementsByTagNameNS(ns, 'TextLine')
            if len(page_lines) > 0:
                this_piece.children = FormatPageUtil.__read_lines(page_lines, this_piece, ns)
            this_piece.parent = top_piece
            region_pieces.append(this_piece)
        top_piece.children = region_pieces
        _all_points = [point for reg in region_pieces for point in reg.dimensions]
        top_piece.dimensions = _all_points
        return top_piece

    @staticmethod
    def __read_lines(page_lines: List[Element], parent, ns) -> List[DigitalObjectTree]:
        line_pieces = []
        for page_line in page_lines:
            line_piece = FormatPageUtil.__from_text_element(page_line, parent, ns)
            word_tokens = page_line.getElementsByTagNameNS(ns, 'Word')
            line_piece.parent = parent
            # inspect PAGE on word level, if set
            if len(word_tokens) > 0:
                try:
                    word_pieces = [FormatPageUtil.__from_text_element(el, line_piece, ns)
                                   for el in word_tokens]
                    if not word_pieces:
                        raise RuntimeError(f"No words in line {line_piece.id}!")
                    # remove line content in favour of words content
                    line_piece.transcriptions = []
                    line_piece.children = word_pieces
                except DigitalObjectException as _pex:
                    raise RuntimeError(_pex.args) from _pex
            line_pieces.append(line_piece)
        return line_pieces

    @staticmethod
    def __from_text_element(element:Element, parent, ns) -> DigitalObjectTree:
        """transformation from PAGE XML textual nodes
        to generic pieces with specific transkription.

        If on PAGE level Word creates word pieces, and
        also inspects textual contents.

        catches several *bad* data flavours regarding
        coordinates, points and text content

        * missing Coord node
        * Coord exists, but misses attribute "points"
        * Coord exists, attribute "points" exists, but
          contains less than 3 point-pairs, thous only
          forms a line and not even a triangle
        * missing TextEquiv node
        * TextEquiv exists, but no Unicode child
        * Unicode exists, but lacking any text content

        """
        element_id = element.getAttribute('id')
        the_level, _local = FormatPageUtil.__map_piece_type(element)
        piece = DigitalObjectTree(
            element_id,
            element,
            file_format=DigitalObjectTreeOCRFileFormat.PAGE
        )
        piece.level = the_level
        piece.parent = parent

        # inspect geometry
        coords = [n for n in element.childNodes if n.localName == 'Coords']
        if len(coords) < 1 or 'points' not in coords[0].attributes:
            raise DigitalObjectException(f"{_local}@ID={element_id} invalid coordinate data")
        first_coord_points = coords[0].getAttribute('points').split()
        # invariant: require at least 3 points, otherwise invalid polygon area
        if len(first_coord_points) < 3:
            raise DigitalObjectException(f"{_local}@ID={element_id} too few points {first_coord_points}")
        try:
            piece.dimensions = [[int(_point.split(',')[0]), int(_point.split(',')[1])]
                                 for _point in first_coord_points]
        except ValueError as _val_err:
            raise DigitalObjectException(f"{_local}@ID={element_id} invalid {first_coord_points}") from _val_err

        # replace current text with next order children text
        # txt_eqs = [n for n in element.childNodes
        #            if n.localName == 'TextEquiv' and \
        #               FormatPageUtil.__contains_textual_content(n)
        #         ]
        txt_eqs = [n for n in element.childNodes if n.localName == 'TextEquiv']
        if txt_eqs:
            first_equiv: Element = txt_eqs[0] # of old
            # if len(txt_eqs) == 1:
            #     first_equiv: Element = txt_eqs[0]
            # else:
            #     # take first indexed element's text content
            #     first_equiv = sorted(txt_eqs, key=lambda x: x.getAttribute('index'))[0]
            unicodes = first_equiv.getElementsByTagNameNS(ns, 'Unicode')
            if len(unicodes) < 1:
                raise DigitalObjectException(f"{_local}@ID={element_id} text missing unicode")
            the_unicode = unicodes[0]
            if the_unicode.firstChild and the_unicode.firstChild.nodeValue is not None:
                # replace linebreak if text only at region level
                txt_content = the_unicode.firstChild.nodeValue.replace('\n', ' ')
                if txt_content:
                    piece.transcription = txt_content
                    # overthrow existing parent transcription
                    if piece.parent is not None and piece.parent.transcriptions:
                        piece.parent.transcriptions = []
        return piece

    @staticmethod
    def __contains_textual_content(element: Element) -> bool:
        unicodes = element.getElementsByTagName('Unicode')
        if len(unicodes) < 1:
            return False
        the_unicode = unicodes[0]
        if the_unicode.firstChild \
            and the_unicode.firstChild.nodeValue is not None \
            and len(the_unicode.firstChild.nodeValue.strip()) > 0:
            return True
        return False

    @staticmethod
    def __map_piece_type(element) -> Tuple[DigitalObjectLevel, str]:
        local_name = element.localName
        the_level = DigitalObjectLevel.UNKNOWN
        if local_name == 'Word':
            the_level = DigitalObjectLevel.WORD
        elif local_name == 'TextLine':
            the_level = DigitalObjectLevel.LINE
        elif local_name == 'TextRegion':
            the_level = DigitalObjectLevel.REGION
        elif local_name == 'TableRegion':
            the_level = DigitalObjectLevel.TABLE
        elif local_name == 'TableCell':
            the_level = DigitalObjectLevel.TABLE_CELL
        return the_level, local_name
