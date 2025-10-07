"""Test specification for reading order in PAGE XML format"""

import pytest

from digital_eval.model.digital_object_model import DigitalObjectTree
from digital_eval.model.main import to_digital_object


def test_reading_order_respected():
    """Test that reading order from RegionRefIndexed is respected
    
    This test uses a file where the reading order differs from
    the natural DOM order of regions. The reading order specifies:
    index 0: region_003, index 1: region_001, index 2: region_002
    while the DOM order is: region_001, region_002, region_003
    """
    
    # arrange - using the test file we created
    import tempfile
    import os
    
    xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
    <Page imageFilename="test.jpg" imageWidth="1000" imageHeight="1000">
        <ReadingOrder>
            <OrderedGroup id="ro_test">
                <RegionRefIndexed index="0" regionRef="region_003"/>
                <RegionRefIndexed index="1" regionRef="region_001"/>
                <RegionRefIndexed index="2" regionRef="region_002"/>
            </OrderedGroup>
        </ReadingOrder>
        <TextRegion id="region_001">
            <Coords points="100,100 200,100 200,200 100,200"/>
            <TextLine id="region_001_line1">
                <Coords points="100,100 200,100 200,150 100,150"/>
                <TextEquiv>
                    <Unicode>Second region text</Unicode>
                </TextEquiv>
            </TextLine>
        </TextRegion>
        <TextRegion id="region_002">
            <Coords points="100,300 200,300 200,400 100,400"/>
            <TextLine id="region_002_line1">
                <Coords points="100,300 200,300 200,350 100,350"/>
                <TextEquiv>
                    <Unicode>Third region text</Unicode>
                </TextEquiv>
            </TextLine>
        </TextRegion>
        <TextRegion id="region_003">
            <Coords points="300,100 400,100 400,200 300,200"/>
            <TextLine id="region_003_line1">
                <Coords points="300,100 400,100 400,150 300,150"/>
                <TextEquiv>
                    <Unicode>First region text</Unicode>
                </TextEquiv>
            </TextLine>
        </TextRegion>
    </Page>
</PcGts>'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(xml_content)
        temp_path = f.name
    
    try:
        # act
        page_piece: DigitalObjectTree = to_digital_object(temp_path)
        
        # assert - regions should be in reading order, not DOM order
        assert len(page_piece.children) == 3
        assert page_piece.children[0].id == 'region_003'
        assert page_piece.children[0].transcription == 'First region text'
        assert page_piece.children[1].id == 'region_001'
        assert page_piece.children[1].transcription == 'Second region text'
        assert page_piece.children[2].id == 'region_002'
        assert page_piece.children[2].transcription == 'Third region text'
    finally:
        os.unlink(temp_path)


def test_reading_order_with_missing_regions():
    """Test that regions without reading order indices are placed after ordered ones"""
    
    import tempfile
    import os
    
    xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
    <Page imageFilename="test.jpg" imageWidth="1000" imageHeight="1000">
        <ReadingOrder>
            <OrderedGroup id="ro_test">
                <RegionRefIndexed index="0" regionRef="region_002"/>
            </OrderedGroup>
        </ReadingOrder>
        <TextRegion id="region_001">
            <Coords points="100,100 200,100 200,200 100,200"/>
            <TextLine id="region_001_line1">
                <Coords points="100,100 200,100 200,150 100,150"/>
                <TextEquiv>
                    <Unicode>No reading order</Unicode>
                </TextEquiv>
            </TextLine>
        </TextRegion>
        <TextRegion id="region_002">
            <Coords points="100,300 200,300 200,400 100,400"/>
            <TextLine id="region_002_line1">
                <Coords points="100,300 200,300 200,350 100,350"/>
                <TextEquiv>
                    <Unicode>Has reading order</Unicode>
                </TextEquiv>
            </TextLine>
        </TextRegion>
    </Page>
</PcGts>'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(xml_content)
        temp_path = f.name
    
    try:
        # act
        page_piece: DigitalObjectTree = to_digital_object(temp_path)
        
        # assert - region_002 (with reading order) should come first
        assert len(page_piece.children) == 2
        assert page_piece.children[0].id == 'region_002'
        assert page_piece.children[0].transcription == 'Has reading order'
        assert page_piece.children[1].id == 'region_001'
        assert page_piece.children[1].transcription == 'No reading order'
    finally:
        os.unlink(temp_path)


def test_no_reading_order_element():
    """Test that regions work correctly when no ReadingOrder element exists"""
    
    import tempfile
    import os
    
    xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
    <Page imageFilename="test.jpg" imageWidth="1000" imageHeight="1000">
        <TextRegion id="region_001">
            <Coords points="100,100 200,100 200,200 100,200"/>
            <TextLine id="region_001_line1">
                <Coords points="100,100 200,100 200,150 100,150"/>
                <TextEquiv>
                    <Unicode>First in DOM</Unicode>
                </TextEquiv>
            </TextLine>
        </TextRegion>
        <TextRegion id="region_002">
            <Coords points="100,300 200,300 200,400 100,400"/>
            <TextLine id="region_002_line1">
                <Coords points="100,300 200,300 200,350 100,350"/>
                <TextEquiv>
                    <Unicode>Second in DOM</Unicode>
                </TextEquiv>
            </TextLine>
        </TextRegion>
    </Page>
</PcGts>'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(xml_content)
        temp_path = f.name
    
    try:
        # act
        page_piece: DigitalObjectTree = to_digital_object(temp_path)
        
        # assert - without reading order, should maintain DOM order
        assert len(page_piece.children) == 2
        assert page_piece.children[0].id == 'region_001'
        assert page_piece.children[0].transcription == 'First in DOM'
        assert page_piece.children[1].id == 'region_002'
        assert page_piece.children[1].transcription == 'Second in DOM'
    finally:
        os.unlink(temp_path)
